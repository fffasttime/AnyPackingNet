import argparse
import time
from typing import Dict, List
import torch
import numpy as np
import mymodel
import sys
import os
from utils.quant_dorefa import activation_quantize_fn

from utils.quant_module import HWGQ, QuantConv2d, ImageInputQ

class ConvParam: ...

def write_hls_config(model_param, path):
    name_mapping = {
        'k': 'K',
        #'s': 'S',
        #'p': 'P',
        'ich': 'IFM_CH',
        'irow': 'IFM_ROL',
        'icol': 'IFM_COL',
        'och': 'OFM_CH',
        'orow': 'OFM_ROW',
        'ocol': 'OFM_COL',
        'abit': 'IN_BIT',
        'wbit': 'W_BIT',
        'incbit': 'INC_BIT',
        'biasbit': 'BIAS_BIT',
        'simd': 'SIMD',
        'pe': 'PE',
        'lshift': 'L_SHIFT'
    }
    content = f'''/********************************************************************************
* Filename: config.h
* Date: {time.ctime()}
* Description: This file is generated by {parser.prog}
*   ptfilename: {opt.filename} 
********************************************************************************/

#ifndef _CONFIG_H_
#define _CONFIG_H_

'''
    for n, conv_param in enumerate(model_param):
        content += f'// conv_{n}\n'
        for k, v in name_mapping.items():
            if hasattr(conv_param, k): # e.g. conv_last has no incbit
                content += f'#define CONV_{n}_{v} {getattr(conv_param, k)}\n'
        content += '\n'
    content += '#endif _CONFIG_H_'

    with open(path + 'config.h', 'w') as f:
        print(content, file=f)

def extract_model(in_shape):
    model_param: List[ConvParam] = []
    feature_map_shape = in_shape
    conv_cnt = 0
    conv_cur = None
    for sub_module in model.modules():
        # expect [QAct] -> [Pooling] -> Conv -> [BN] -> [Pooling], state machine mode
        if isinstance(sub_module, HWGQ) or isinstance(sub_module, ImageInputQ) or isinstance(sub_module, activation_quantize_fn):
            print('  Detected ActQ Layer', end='')
            if conv_cur is None: conv_cur = ConvParam()
            if isinstance(sub_module, HWGQ) or isinstance(sub_module, ImageInputQ):
                conv_cur.abit = sub_module.bit
                conv_cur.astep = sub_module.step
            else:
                conv_cur.abit = sub_module.a_bit
                conv_cur.astep = 1/2**conv_cur.abit
            
            conv_cur.actq_class = type(sub_module).__name__
            print(f', abit {conv_cur.abit}, astep {conv_cur.astep}, class {conv_cur.actq_class}')

            if conv_cnt: # previous.obit = cur.abit
                model_param[conv_cnt-1].obit = conv_cur.abit
                model_param[conv_cnt-1].ostep = conv_cur.astep
            
        elif isinstance(sub_module, torch.nn.Conv2d):
            if conv_cur is None: conv_cur = ConvParam()
            conv_cur.n = conv_cnt
            print('Extract conv_%d'%conv_cnt, end='')

            conv_cur.k = sub_module.kernel_size[0]
            conv_cur.s = sub_module.stride[0]
            conv_cur.p = sub_module.padding[0]
            conv_cur.ich = sub_module.in_channels
            conv_cur.och = sub_module.out_channels
            conv_cur.irow = feature_map_shape[1]
            conv_cur.icol = feature_map_shape[2]
            
            feature_map_shape[0] = sub_module.out_channels
            feature_map_shape[1] = (feature_map_shape[1] + 2 * sub_module.padding[0] - sub_module.kernel_size[0]) // sub_module.stride[0] + 1
            feature_map_shape[2] = (feature_map_shape[2] + 2 * sub_module.padding[0] - sub_module.kernel_size[0]) // sub_module.stride[0] + 1
            conv_cur.orow = feature_map_shape[1]
            conv_cur.ocol = feature_map_shape[2]

            if sub_module.bias is not None:
                conv_cur.convbias = sub_module.bias.detach().numpy()
                print(', +bias', end='')

            if isinstance(sub_module, QuantConv2d): # New quant
                conv_cur.wbit = sub_module.bit
                conv_cur.w, conv_cur.wstep = sub_module.export_quant() # wstep is not QuantConv2d.step becuause of alpha

            elif type(sub_module).__name__ == 'Conv2d_Q': # Old dorefa quant
                conv_cur.wbit = sub_module.w_bit
                conv_cur.wstep = 1/2**(conv_cur.wbit-1)
                weight = np.tanh(sub_module.weight.detach().numpy())
                weight = weight / np.max(np.abs(weight))
                n = 2**(conv_cur.wbit-1)
                weight_q = weight * n
                weight_q = np.clip(np.round(weight_q),-n, n-1)
                weight_q = weight_q.astype(np.int32)
                conv_cur.w = weight_q
            else:
                raise NotImplementedError(sub_module)
            print(', ich {ich}, och {och}, irow {irow}, icol {icol}, ksp {k}{s}{p}, wbit {wbit}, wstep {wstep}'.format(**vars(conv_cur)))
            
            model_param.append(conv_cur)
            conv_cur = None
            conv_cnt += 1
        
        elif isinstance(sub_module, torch.nn.BatchNorm2d):
            print('  Detected BatchNorm2d')
            gamma = sub_module.weight
            beta = sub_module.bias
            mean = sub_module.running_mean
            var = sub_module.running_var
            eps = sub_module.eps
            
            model_param[-1].bn_w = (gamma / (torch.sqrt(var) + eps)).detach().numpy()
            model_param[-1].bn_b = (beta - (mean / (torch.sqrt(var) + eps) * gamma)).detach().numpy()

        elif isinstance(sub_module, torch.nn.MaxPool2d):
            feature_map_shape[1] = feature_map_shape[1] // sub_module.kernel_size
            feature_map_shape[2] = feature_map_shape[2] // sub_module.kernel_size
    
    if not hasattr(model_param[0], 'abit'): # train code rescaled [0,255] to [0,1) by /256 default
        model_param[0].abit = 8
    if not hasattr(model_param[0], 'astep'):
        model_param[0].astep = 1/256

    return model_param

def process_batchnorm(model_param):
    '''process_batchnorm(model_param)
    Merge wstep, astep, ostep scale into batchnorm, then quantize. 

    Method:
    Define MAC = Conv(w, a), out = MAC*BN_w + BN_b,
    wq = w/wstep, aq = a/astep, MACq = MAC/MACstep, outq = out/ostep.

    outq = (MAC*BN_w + BN_b) / ostep
         = MACq * (MACstep/ostep)*BN_w + BN_b/ostep
         = MACq *     inc_raw          + bias_raw
    next layer activation a' = ActQ(out), i.e. a'q = clip(round(outq))

    Quantiaztion of inc_raw & bias_raw: 
    outq_real = (MACq*round(inc_raw*scale) + round(bias_raw*scale)) // scale  ; where scale=2**T
              = (MACq*        inc          +         bias         ) >> T

    Params:
    T = (wbit-1)+abit+lshift  # This comes from dorefa quant, not optimal
    MBIT = wbit+abit+ceil(log2(sum_number))
    incbit = len(bit(inc)); biasbit = len(bit(bias))
    larger lshift is better, but MBIT+incbit<48
    '''
    lshift = 16

    for conv in model_param[:-1]:
        print(f'Process bn_{conv.n}, shape {conv.bn_w.shape},', end = ' ')

        # Merge step to BN
        conv.lshift = lshift
        MACstep = conv.wstep * conv.astep
        ostep = conv.ostep
        inc_raw = conv.bn_w * MACstep / ostep
        bias_raw = conv.bn_b / ostep

        # Quantization
        T = lshift+conv.wbit+conv.abit-1
        conv.inc = np.round(inc_raw * 2**T).astype(np.int64)
        conv.bias = np.round(bias_raw * 2**T).astype(np.int64)
        conv.lshift_T = T
        # Get bitlength
        bitlength = lambda x: 1 + int(np.abs(x).max()).bit_length()
        conv.incbit = bitlength(conv.inc)
        conv.biasbit = bitlength(conv.bias)
        print(f'incbit {conv.incbit}, biasbit {conv.biasbit}')
    
    conv_last = model_param[-1] # process lastbias
    conv_last.inc = None
    conv_last.div = 1/(conv.wstep * conv.astep)
    conv_last.bias = np.round(conv_last.convbias * conv_last.div).astype(np.int64)
    conv_last.biasbit = bitlength(conv_last.bias)
    print(f'conv_last biasbit {conv_last.biasbit}, div {conv_last.div}')

def reorder_weight(model_param, layers_simd, layers_pe):
    '''reorder_weight(model_param)
    Reorder array for hlscode.
    '''

    for conv, simd, pe in zip(model_param, layers_simd, layers_pe):
        print(f'Reorder conv_{conv.n}, w {conv.w.shape}', end='')
        conv.simd = simd
        conv.pe = pe

        # process batchnorm
        if conv.inc is not None:
            conv.inc = conv.inc.reshape(conv.och//conv.pe, conv.pe).T
        if conv.bias is not None:
            conv.bias = conv.bias.reshape(conv.och//conv.pe, conv.pe).T
        
        # process conv weight
        w = conv.w    # [och, ich, kr, kc]
        assert conv.och%conv.pe == 0, f"conv_{conv.n}, och {conv.och}, pe {conv.pe}"
        assert conv.k*conv.ich%simd == 0, f"conv_{conv.n}, ich {conv.ich}, k {conv.k}, simd {conv.simd}"

        if conv.n==0: # first layer is different
            w = w.transpose(0, 2, 3, 1) # [och, kr, kc, ich]
        else:
            w = w.transpose(0, 3, 2, 1) # [och, kc, kr, ich]

        w = w.reshape(conv.och//conv.pe, conv.pe, conv.k, conv.k*conv.ich//simd, simd)
        w = w.transpose(1,2,0,3,4) # [pe, k, och/pe, k*ich/simd, simd]
        w = w.reshape(conv.pe, conv.k, -1, simd) # hls format [pe, k, och/pe*k*ich/simd, simd]

        if conv.k == 1: # kernel size=1
            w = w.reshape(conv.pe, -1, simd)
        print(' ->', w.shape)

        conv.w = w

def print_ndarray_recursion(arr, str_func=str, file=sys.stdout, stop=0):
    if not hasattr(arr, '__iter__') or len(arr.shape) == stop:
        print(str_func(arr), file=file, end='')
        return
    ends = '' if (len(arr.shape)==stop+1) else '\n'
    print('{', file=file, end='')
    for i, item in enumerate(arr):
        print_ndarray_recursion(item, str_func, file, stop)
        if i!=len(arr)-1: print(',', file=file, end=ends)
    print(ends+'}', file=file, end='')

def write_hls_weights(model_param, path):
    '''write_hls_weights(model_param, path)
    Write hls weights+inc+bias array code according to numpy shape.
    '''
    f = open(path + 'weights.hpp', 'w')

    print(f'''/********************************************************************************
* Filename: weights.hpp
* Date: {time.ctime()}
* Description: This file is generated by {parser.prog}
*   ptfilename: {opt.filename} 
********************************************************************************/

#ifndef _WEIGHTS_HPP_
#define _WEIGHTS_HPP_
#include <ap_int.h>
''', file=f)

    for conv in model_param:
        n = conv.n
        print(f"Write conv_{n} weight, pe {conv.pe}, simd {conv.simd}, wbit {conv.wbit}")
        print(f"// layer: {n}, PE: {conv.pe}, SIMD: {conv.simd}, wbit: {conv.wbit}", file=f)

        # print conv weight,  merge [SIMD] value into one ap_uint
        if conv.k>1:
            print(f"const ap_uint<{conv.wbit * conv.simd}> conv_{n}_w[{conv.pe}][{conv.k}][{conv.w.shape[2]}]=", file=f)
        else:
            print(f"const ap_uint<{conv.wbit * conv.simd}> conv_{n}_w[{conv.pe}][{conv.w.shape[1]}]=", file=f)
        hex_str = lambda x: '"' + hex(x) + '"'
        def pack1d_str(arr): # x: 1d-array
            x = 0
            for v in arr[::-1]: # [!] reverse simd pack, it is related to hls implemention
                v = int(v) # use python bignumber, not np.int
                assert -1<<conv.wbit-1 <= v < 1<<conv.wbit-1, f'got v={v} while wbit={conv.wbit}'
                x=(x<<conv.wbit) + (v&(2**conv.wbit-1))
            return hex_str(x)
        print_ndarray_recursion(conv.w, pack1d_str, f, stop=1)
        print(';', file=f)

        # print inc, bias
        if conv.inc is not None:
            print(f"const ap_int<{conv.incbit}> conv_{n}_inc[{conv.pe}][{conv.och//conv.pe}]=", file=f)
            print_ndarray_recursion(conv.inc, hex_str, f)
            print(';', file=f)
        if conv.bias is not None:
            print(f"const ap_int<{conv.biasbit}> conv_{n}_bias[{conv.pe}][{conv.och//conv.pe}]=", file=f)
            print_ndarray_recursion(conv.bias, hex_str, f)
            print(';', file=f)
    
    print('#endif', file=f)
    f.close()


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('filename', help='.pt file name in ./weights/')
    parser.add_argument('-m', '--model', default='UltraNet_FixQ', help = 'model class name in mymodel.py')
    parser.add_argument('-c', '--config-simd-pe', default='config_simd_pe', help = '.txt file in ./hls/')
    opt = parser.parse_args()
    simd_pe = np.loadtxt('hls/'+opt.config_simd_pe+'.txt', dtype=int, skiprows=1)
    dir_output = 'hls/' + opt.filename.split('.')[0] + '/'
    if not os.path.exists(dir_output): os.makedirs(dir_output)

    # load model and state_dict
    ptfile:Dict = torch.load('weights/' + opt.filename + '.pt', map_location='cpu')
    model = getattr(mymodel, opt.model)(**ptfile.setdefault('model_params', {}))
    model.load_state_dict(ptfile['model'])

    # processs
    model_param = extract_model([1, 160, 320])
    process_batchnorm(model_param) # get bn param before write hls config
    torch.save(model_param, dir_output + 'model_param.pkl')
    reorder_weight(model_param, simd_pe[:,0], simd_pe[:,1]) # get pe, simd param before write hls config
    write_hls_config(model_param, dir_output)
    write_hls_weights(model_param, dir_output)
