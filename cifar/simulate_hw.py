import argparse
import torch
import torch.nn.functional as F
import numpy as np
import torch.nn as nn

from export_hls import ConvParam
from test_acc import testset
from utils.view_pt import select_weight_file

class QConvLayer:
    def __init__(self, conv_param):
        self.conv = conv_param
        self.w = torch.tensor(self.conv.w, dtype = torch.int64)
    
    def __call__(self, x: torch.Tensor, downsampling):
        if self.conv.icol < x.shape[-1]: # Maxpool. Note: Order of Maxpool and BN is IMPORTANT when BN.inc can be negative
            assert self.conv.irow*2, self.conv.icol*2 == x.shape[2:]
            x = F.max_pool2d(x.float(), kernel_size = 2, stride = 2).to(dtype=torch.int64)

        if self.conv.type == 'linear':
            x = x.flatten(1)
            x = F.linear(x, self.w)
            x += self.conv.bias
            return x
        # print('convi', self.conv.n, x[0,0,:,:])

        x = F.conv2d(x, self.w, bias=None, stride=self.conv.s, padding=self.conv.p) # [N, OCH, OROW, OCOL]
        # print('convo', self.conv.n, x[0,0,:,:])
        #if downsampling: # Maxpool
        #    x = F.max_pool2d(x.float(), kernel_size = 2, stride = 2).to(dtype=torch.int64)
        och = x.shape[1]
        if True:
            if self.conv.inc is not None:
                inc_ch = self.conv.inc.reshape((1, och, 1, 1))
                x *= inc_ch
            if hasattr(self.conv, 'bias'):
                bias_ch = self.conv.bias.reshape((1, och, 1, 1))
                x += bias_ch

            # print('biaso', self.conv.n, x[0,0,:,:]/2**self.conv.lshift_T)
            if hasattr(self.conv, 'lshift'):
                x += 1 << self.conv.lshift_T-1
                x >>= self.conv.lshift_T

        else: ## no inc/bias quantization
            if self.conv.inc is not None:
                inc_ch = self.conv.inc_raw.reshape((1, och, 1, 1))
                x *= inc_ch
            if hasattr(self.conv, 'bias'):
                bias_ch = self.conv.bias_raw.reshape((1, och, 1, 1))
                x += bias_ch
            # print('biaso', self.conv.n, x[0,0,:,:])
            x = torch.round(x).to(dtype = torch.int64)
        
        if hasattr(self.conv, 'obit'):
            x.clip_(0, 2**(self.conv.obit)-1)
        return x

class HWModel:
    def __init__(self, model_param):
        self.layers = [QConvLayer(conv_param) for conv_param in model_param]

    def __call__(self, x):
        assert len(x.shape) == 4 and x.dtype == torch.int64
        img_size = x.shape[-2:]

        if self.layers[0].conv.abit<8: # ImageInputQ
            x=x>>(8-self.layers[0].conv.abit) 

        for i, layer in enumerate(self.layers):
            x = layer(x, self.layers[i+1].conv.icol<layer.conv.icol if i+1<len(self.layers) else False)
        
        x = x.float() / self.layers[-1].conv.div
        return x

def testdataset(hwmodel):
    testloader = torch.utils.data.DataLoader(testset, batch_size=opt.batch_size, shuffle=False, num_workers=2)
    criterion = nn.CrossEntropyLoss()

    mloss = macc = 0.
    for i, (inputs, labels) in enumerate(testloader):
        if i == opt.num_batch: break
        bn, _, height, width = inputs.shape  # batch size, channels, height, width

        inputs *= 256.0
        inputs = inputs.to(dtype = torch.int64)
        inf_out = hwmodel(inputs)
        _, predicted = torch.max(inf_out.data, 1)
        correct = (predicted == labels).sum().item()
        loss = criterion(inf_out, labels)

        np.set_printoptions(precision = 2)
        for p in range(len(inputs)):
            print(predicted[p].numpy(), labels[p].numpy(), inf_out[p].numpy())

        mloss = (mloss*i + loss.item()) / (i+1)
        macc = (macc*i + correct/opt.batch_size) / (i+1)

    print('acc %.2f, loss %.4f'%(macc*100, mloss))

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-w', '--weight', help='weight folder name in ./hls/, which contians model_param.pkl')
    parser.add_argument('-bs', '--batch-size', type=int, default=1, help = 'batch-size')
    parser.add_argument('-nb', '--num-batch', type=int, default=1, help = 'num of batchs to run, -1 for full dataset')
    opt = parser.parse_args()
    print(opt)
    if opt.weight is None: opt.weight = select_weight_file()
    
    x = torch.zeros([1,3,32,32], dtype=torch.int64)
    hwmodel = HWModel(torch.load('hls/'+opt.weight+'/model_param.pkl'))
    
    testdataset(hwmodel)
