from __future__ import print_function
from turtle import forward
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

gaussian_steps = {1: 1.596, 2: 0.996, 3: 0.586, 4: 0.336, 5: 0.190, 6: 0.106, 7: 0.059, 8: 0.032}
hwgq_steps = {1: 0.799, 2: 0.538, 3: 0.3217, 4: 0.185, 5: 0.104, 6: 0.058, 7: 0.033, 8: 0.019}


dsp_factors_k11=[
[12,8,8,6,6,4,4],
[10,8,6,6,4,4,4],
[8,6,6,4,4,4,3],
[6,6,4,4,4,4,2],
[6,4,4,4,2,2,2],
[4,4,4,4,2,2,2],
[4,4,3,2,2,2,2],
]

dsp_factors_k33=[
[18,15,12,7.5,7.5,6,6],
[15,12,7.5,6,6,6,3],
[12,7.5,6,6,6,6,3],
[9,6,6,6,6,3,3],
[7.5,6,6,4.5,3,3,3],
[6,6,4.5,3,3,3,2.25],
[6,3,3,3,3,3,2],
]

dsp_factors_k55=[
[20,15,10,7.5,7.5,5,5],
[12.5,10,6.67,5,5,5,3.33],
[10,7.5,5,5,5,5,3.33],
[7.5,6.67,5,5,5,3.33,3.33],
[6.67,5,5,5,3.33,2.5,2.5],
[5,5,5,3.33,2.5,2.5,2.5],
[5,3.33,3.33,3.33,2.5,2.5,2],
]

class _gauss_quantize_sym(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, step, bit):
        lvls = 2 ** bit / 2
        alpha = x.std().item()
        step *= alpha
        y = (torch.round(x/step+0.5)-0.5) * step
        thr = (lvls-0.5)*step
        y = y.clamp(min=-thr, max=thr)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None, None


class _gauss_quantize_resclaed_step_sym(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, step, bit):
        lvls = 2 ** bit / 2
        y = (torch.round(x/step+0.5)-0.5) * step
        thr = (lvls-0.5)*step
        y = y.clamp(min=-thr, max=thr)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None, None


class _gauss_quantize(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, step, bit):
        lvls = 2 ** bit / 2
        alpha = x.std().item()
        step *= alpha
        y = torch.clamp(torch.round(x/step), -lvls, lvls-1) * step
        return y

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None, None

def _gauss_quantize_export(x, step, bit):
    lvls = 2 ** bit / 2
    alpha = x.std().item()
    step *= alpha
    y = torch.clamp(torch.round(x/step), -lvls, lvls-1)
    return y.cpu().detach().int().numpy(), step

class _gauss_quantize_resclaed_step(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, step, bit):
        lvls = 2 ** bit / 2
        y = torch.clamp(torch.round(x/step), -lvls, lvls-1) * step
        return y

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None, None

class _hwgq(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, step):
        y = torch.round(x / step) * step
        return y

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None


class HWGQ(nn.Module):
    def __init__(self, bit=2):
        super(HWGQ, self).__init__()
        self.bit = bit
        if bit < 32:
            self.step = hwgq_steps[bit]
        else:
            self.step = None

    def forward(self, x):
        if self.bit >= 32:
            return x.clamp(min=0.0)
        lvls = float(2 ** self.bit - 1)
        clip_thr = self.step * lvls
        y = x.clamp(min=0.0, max=clip_thr)
        out = _hwgq.apply(y, self.step)
        return out

class ImageInputQ(nn.Module):
    '''
    Assume image input are discrete value [0/256, 1/256, 2/256, ..., 255/256]
    '''
    def __init__(self, bit = 8):
        super(ImageInputQ, self).__init__()
        self.bit = bit
        self.step = 1/2**bit

    def forward(self, x):
        if self.step==32:
            return out
        out = torch.floor(x/self.step) * self.step  # [!] There will be no gradient on x
        return out

class QuantConv2d(nn.Conv2d):

    def __init__(self, *kargs, **kwargs):
        self.bit = kwargs.pop('bit', 1)
        super(QuantConv2d, self).__init__(*kargs, **kwargs)
        assert self.bit > 0
        self.step = None if self.bit==32 else gaussian_steps[self.bit]

    def forward(self, input):
        # quantized conv, otherwise regular
        if self.bit < 32:
            quant_weight = _gauss_quantize.apply(self.weight, self.step, self.bit)
            out = F.conv2d(
                input, quant_weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        else:
            out = F.conv2d(
                input, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        return out

    def export_quant(self):
        return _gauss_quantize_export(self.weight, self.step, self.bit)

class QuantLinear(nn.Linear):

    def __init__(self, *kargs, **kwargs):
        self.bit = kwargs.pop('bit', 1)
        super(QuantLinear, self).__init__(*kargs, **kwargs)
        assert self.bit > 0
        self.step = gaussian_steps[self.bit]

    def forward(self, input):
        # quantized linear, otherwise regular
        if self.bit < 32:
            # assert self.bias is None
            quant_weight = _gauss_quantize.apply(self.weight, self.step, self.bit)
            out = F.linear(input, quant_weight, self.bias)
        else:
            out = F.linear(input, self.weight, self.bias)
        return out

    def export_quant(self):
        return _gauss_quantize_export(self.weight, self.step, self.bit)

class QuantActivConv2d(nn.Module):

    def __init__(self, inplane, outplane, wbit=1, abit=2, ActQ = HWGQ, **kwargs):
        super(QuantActivConv2d, self).__init__()
        self.abit = abit
        self.wbit = wbit
        self.activ = ActQ(abit)
        self.conv = QuantConv2d(inplane, outplane, bit=wbit, **kwargs)
        # complexities
        stride = kwargs['stride'] if 'stride' in kwargs else 1
        if isinstance(kwargs['kernel_size'], tuple):
            kernel_size = kwargs['kernel_size'][0] * kwargs['kernel_size'][1]
        else:
            kernel_size = kwargs['kernel_size'] * kwargs['kernel_size']
        self.kernel_size = kwargs['kernel_size']
        if 'groups' in kwargs: groups = kwargs['groups']
        else: groups = 1
        self.param_size = inplane * outplane * kernel_size * 1e-6 / groups
        self.filter_size = self.param_size / float(stride ** 2.0)
        self.register_buffer('size_product', torch.tensor(0, dtype=torch.float))
        self.register_buffer('memory_size', torch.tensor(0, dtype=torch.float))

    def forward(self, input):
        in_shape = input.shape
        tmp = torch.tensor(in_shape[1] * in_shape[2] * in_shape[3] * 1e-3, dtype=torch.float)
        self.memory_size.copy_(tmp)
        tmp = torch.tensor(self.filter_size * in_shape[-1] * in_shape[-2], dtype=torch.float)
        self.size_product.copy_(tmp)
        out = self.activ(input)
        ## print('ii',input[0,0,:,0]/self.activ.step)
        ## print('convi', torch.round(out[0,0,:,0]/self.activ.step).int())
        ## wstd = self.conv.weight.std()
        out = self.conv(out)
        ## print('convo', torch.round(out[0,0,:,0]/(self.activ.step*self.conv.step*wstd)).int())
        return out


class QuantActivLinear(nn.Module):

    def __init__(self, inplane, outplane, wbit=1, abit=2, **kwargs):
        super(QuantActivLinear, self).__init__()
        self.abit = abit
        self.wbit = wbit
        self.activ = HWGQ(abit)
        self.linear = QuantLinear(inplane, outplane, bit=wbit, **kwargs)
        # complexities
        self.param_size = inplane * outplane * 1e-6
        self.register_buffer('size_product', torch.tensor(self.param_size, dtype=torch.float))
        self.register_buffer('memory_size', torch.tensor(0, dtype=torch.float))

    def forward(self, input):
        tmp = torch.tensor(input.shape[1] * 1e-3, dtype=torch.float)
        self.memory_size.copy_(tmp)
        out = self.activ(input)
        out = self.linear(out)
        return out


class MixQuantActiv(nn.Module):

    def __init__(self, bits, ActQ = HWGQ):
        super(MixQuantActiv, self).__init__()
        self.bits = bits
        self.alpha_activ = Parameter(torch.Tensor(len(self.bits)))
        self.alpha_activ.data.fill_(0.01)
        self.mix_activ = nn.ModuleList()
        for bit in self.bits:
            self.mix_activ.append(ActQ(bit=bit))

    def forward(self, input):
        outs = []
        sw = F.softmax(self.alpha_activ, dim=0)
        for i, branch in enumerate(self.mix_activ):
            outs.append(branch(input) * sw[i])
        activ = sum(outs)
        return activ


class MixQuantConv2d(nn.Module):

    def __init__(self, inplane, outplane, bits, **kwargs):
        super(MixQuantConv2d, self).__init__()
        assert not kwargs['bias']
        self.bits = bits
        self.alpha_weight = Parameter(torch.Tensor(len(self.bits)))
        self.alpha_weight.data.fill_(0.01)
        self.conv_list = nn.ModuleList()
        self.steps = []
        for bit in self.bits:
            assert 0 < bit < 32
            self.conv_list.append(nn.Conv2d(inplane, outplane, **kwargs))
            self.steps.append(gaussian_steps[bit])

    def forward(self, input):
        mix_quant_weight = []
        sw = F.softmax(self.alpha_weight, dim=0)
        for i, bit in enumerate(self.bits):
            weight = self.conv_list[i].weight
            weight_std = weight.std().item()
            step = self.steps[i] * weight_std
            quant_weight = _gauss_quantize_resclaed_step.apply(weight, step, bit)
            scaled_quant_weight = quant_weight * sw[i]
            mix_quant_weight.append(scaled_quant_weight)
        mix_quant_weight = sum(mix_quant_weight)
        conv = self.conv_list[0]
        out = F.conv2d(
            input, mix_quant_weight, conv.bias, conv.stride, conv.padding, conv.dilation, conv.groups)
        return out


class SharedMixQuantConv2d(nn.Module):

    def __init__(self, inplane, outplane, bits, **kwargs):
        super(SharedMixQuantConv2d, self).__init__()
        # assert not kwargs['bias']
        self.bits = bits
        self.alpha_weight = Parameter(torch.Tensor(len(self.bits)))
        self.alpha_weight.data.fill_(0.01)
        self.conv = nn.Conv2d(inplane, outplane, **kwargs)
        self.steps = []
        for bit in self.bits:
            assert 0 < bit < 32
            self.steps.append(gaussian_steps[bit])

    def forward(self, input):
        mix_quant_weight = []
        sw = F.softmax(self.alpha_weight, dim=0)
        conv = self.conv
        weight = conv.weight
        # save repeated std computation for shared weights
        weight_std = weight.std().item()
        for i, bit in enumerate(self.bits):
            step = self.steps[i] * weight_std
            quant_weight = _gauss_quantize_resclaed_step.apply(weight, step, bit)
            scaled_quant_weight = quant_weight * sw[i]
            mix_quant_weight.append(scaled_quant_weight)
        mix_quant_weight = sum(mix_quant_weight)
        out = F.conv2d(
            input, mix_quant_weight, conv.bias, conv.stride, conv.padding, conv.dilation, conv.groups)
        return out


class MixActivConv2d(nn.Module):

    def __init__(self, inplane, outplane, wbits=None, abits=None, share_weight=False, ActQ = HWGQ, **kwargs):
        super(MixActivConv2d, self).__init__()
        if wbits is None:
            self.wbits = [1, 2]
        else:
            self.wbits = wbits
        if abits is None:
            self.abits = [1, 2]
        else:
            self.abits = abits
        # build mix-precision branches
        self.mix_activ = MixQuantActiv(self.abits, ActQ = ActQ)
        self.share_weight = share_weight
        if share_weight:
            self.mix_weight = SharedMixQuantConv2d(inplane, outplane, self.wbits, **kwargs)
        else:
            self.mix_weight = MixQuantConv2d(inplane, outplane, self.wbits, **kwargs)
        # complexities
        stride = kwargs['stride'] if 'stride' in kwargs else 1
        if isinstance(kwargs['kernel_size'], tuple):
            kernel_size = kwargs['kernel_size'][0] * kwargs['kernel_size'][1]
        else:
            kernel_size = kwargs['kernel_size'] * kwargs['kernel_size']
        self.kernel_size = kwargs['kernel_size']
        
        if 'groups' in kwargs: groups = kwargs['groups']
        else: groups = 1
        self.param_size = inplane * outplane * kernel_size * 1e-6 / groups
        self.filter_size = self.param_size / float(stride ** 2.0)
        self.register_buffer('size_product', torch.tensor(0, dtype=torch.float))
        self.register_buffer('memory_size', torch.tensor(0, dtype=torch.float))

    def forward(self, input):
        in_shape = input.shape
        tmp = torch.tensor(in_shape[1] * in_shape[2] * in_shape[3] * 1e-3, dtype=torch.float)
        self.memory_size.copy_(tmp)
        tmp = torch.tensor(self.filter_size * in_shape[-1] * in_shape[-2], dtype=torch.float)
        self.size_product.copy_(tmp)
        out = self.mix_activ(input)
        out = self.mix_weight(out)
        return out

    def complexity_loss_old(self):
        sw = F.softmax(self.mix_activ.alpha_activ, dim=0)
        mix_abit = 0
        abits = self.mix_activ.bits
        for i in range(len(abits)):
            mix_abit += sw[i] * abits[i]
        sw = F.softmax(self.mix_weight.alpha_weight, dim=0)
        mix_wbit = 0
        wbits = self.mix_weight.bits
        for i in range(len(wbits)):
            mix_wbit += sw[i] * wbits[i]
        complexity = self.size_product.item() * mix_abit * mix_wbit
        return complexity
    
    def complexity_loss(self):
        sa = F.softmax(self.mix_activ.alpha_activ, dim=0)
        abits = self.mix_activ.bits
        sw = F.softmax(self.mix_weight.alpha_weight, dim=0)
        mix_scale = 0
        wbits = self.mix_weight.bits

        if self.kernel_size == 1:
            dsp_factors = dsp_factors_k11
        elif self.kernel_size == 3:
            dsp_factors = dsp_factors_k33
        elif self.kernel_size == 5:
            dsp_factors = dsp_factors_k55
        else:
            raise NotImplementedError
        for i in range(len(wbits)):
            for j in range(len(abits)):
                mix_scale += sw[i] * sa[j] / dsp_factors[wbits[i]-2][abits[j]-2]
        complexity = self.size_product.item() * 64 * mix_scale
        return complexity


    def fetch_best_arch(self, layer_idx):
        size_product = float(self.size_product.cpu().numpy())
        memory_size = float(self.memory_size.cpu().numpy())
        prob_activ = F.softmax(self.mix_activ.alpha_activ, dim=0)
        prob_activ = prob_activ.detach().cpu().numpy()
        best_activ = prob_activ.argmax()
        mix_abit = 0
        abits = self.mix_activ.bits
        for i in range(len(abits)):
            mix_abit += prob_activ[i] * abits[i]
        prob_weight = F.softmax(self.mix_weight.alpha_weight, dim=0)
        prob_weight = prob_weight.detach().cpu().numpy()
        best_weight = prob_weight.argmax()
        mix_wbit = 0
        wbits = self.mix_weight.bits
        for i in range(len(wbits)):
            mix_wbit += prob_weight[i] * wbits[i]
        if self.share_weight:
            weight_shape = list(self.mix_weight.conv.weight.shape)
        else:
            weight_shape = list(self.mix_weight.conv_list[0].weight.shape)
        print('idx {} with shape {}, activ alpha: {}, comp: {:.3f}M * {:.3f} * {:.3f}, '
              'memory: {:.3f}K * {:.3f}'.format(layer_idx, weight_shape, prob_activ, size_product,
                                                mix_abit, mix_wbit, memory_size, mix_abit))
        print('idx {} with shape {}, weight alpha: {}, comp: {:.3f}M * {:.3f} * {:.3f}, '
              'param: {:.3f}M * {:.3f}'.format(layer_idx, weight_shape, prob_weight, size_product,
                                               mix_abit, mix_wbit, self.param_size, mix_wbit))
        best_arch = {'best_activ': [best_activ], 'best_weight': [best_weight]}
        bitops = size_product * abits[best_activ] * wbits[best_weight]
        bita = memory_size * abits[best_activ]
        bitw = self.param_size * wbits[best_weight]
        
        if self.kernel_size == 1:
            dsp_factors = dsp_factors_k11
        elif self.kernel_size == 3:
            dsp_factors = dsp_factors_k33
        elif self.kernel_size == 5:
            dsp_factors = dsp_factors_k55
        else:
            raise NotImplementedError
        dsps = size_product / dsp_factors[wbits[best_weight]-2][abits[best_activ]-2]
        mixbitops = size_product * mix_abit * mix_wbit
        mixbita = memory_size * mix_abit
        mixbitw = self.param_size * mix_wbit
        mixdsps = 0
        for i in range(len(wbits)):
            for j in range(len(abits)):
                mixdsps += prob_weight[i] * prob_activ[j] / dsp_factors[wbits[i]-2][abits[j]-2]
        mixdsps *= size_product

        return best_arch, bitops, bita, bitw, mixbitops, mixbita, mixbitw, dsps, mixdsps


class SharedMixQuantLinear(nn.Module):

    def __init__(self, inplane, outplane, bits, **kwargs):
        super(SharedMixQuantLinear, self).__init__()
        # assert not kwargs['bias']
        self.bits = bits
        self.alpha_weight = Parameter(torch.Tensor(len(self.bits)))
        self.alpha_weight.data.fill_(0.01)
        self.linear = nn.Linear(inplane, outplane, **kwargs)
        self.steps = []
        for bit in self.bits:
            assert 0 < bit < 32
            self.steps.append(gaussian_steps[bit])

    def forward(self, input):
        mix_quant_weight = []
        sw = F.softmax(self.alpha_weight, dim=0)
        linear = self.linear
        weight = linear.weight
        # save repeated std computation for shared weights
        weight_std = weight.std().item()
        for i, bit in enumerate(self.bits):
            step = self.steps[i] * weight_std
            quant_weight = _gauss_quantize_resclaed_step.apply(weight, step, bit)
            scaled_quant_weight = quant_weight * sw[i]
            mix_quant_weight.append(scaled_quant_weight)
        mix_quant_weight = sum(mix_quant_weight)
        out = F.linear(input, mix_quant_weight, linear.bias)
        return out

class MixActivLinear(nn.Module):
    def __init__(self, inplane, outplane, wbits=None, abits=None, share_weight=True, **kwargs):
        super(MixActivLinear, self).__init__()
        if wbits is None:
            self.wbits = [1, 2]
        else:
            self.wbits = wbits
        if abits is None:
            self.abits = [1, 2]
        else:
            self.abits = abits
        # build mix-precision branches
        self.mix_activ = MixQuantActiv(self.abits)
        assert share_weight
        self.share_weight = share_weight
        self.mix_weight = SharedMixQuantLinear(inplane, outplane, self.wbits, **kwargs)
        # complexities
        self.param_size = inplane * outplane * 1e-6
        self.register_buffer('size_product', torch.tensor(self.param_size, dtype=torch.float))
        self.register_buffer('memory_size', torch.tensor(0, dtype=torch.float))

    def forward(self, input):
        tmp = torch.tensor(input.shape[1] * 1e-3, dtype=torch.float)
        self.memory_size.copy_(tmp)
        out = self.mix_activ(input)
        out = self.mix_weight(out)
        return out

    def complexity_loss_old(self):
        sw = F.softmax(self.mix_activ.alpha_activ, dim=0)
        mix_abit = 0
        abits = self.mix_activ.bits
        for i in range(len(abits)):
            mix_abit += sw[i] * abits[i]
        sw = F.softmax(self.mix_weight.alpha_weight, dim=0)
        mix_wbit = 0
        wbits = self.mix_weight.bits
        for i in range(len(wbits)):
            mix_wbit += sw[i] * wbits[i]
        complexity = self.size_product.item() * mix_abit * mix_wbit
        return complexity

    def complexity_loss(self):
        sa = F.softmax(self.mix_activ.alpha_activ, dim=0)
        abits = self.mix_activ.bits
        sw = F.softmax(self.mix_weight.alpha_weight, dim=0)
        mix_scale = 0
        wbits = self.mix_weight.bits
        for i in range(len(wbits)):
            for j in range(len(abits)):
                mix_scale += sw[i] * sa[j] / dsp_factors_k11[wbits[i]-2][abits[j]-2]
        complexity = self.size_product.item() * 64 * mix_scale
        return complexity

    def fetch_best_arch(self, layer_idx):
        size_product = float(self.size_product.cpu().numpy())
        memory_size = float(self.memory_size.cpu().numpy())
        prob_activ = F.softmax(self.mix_activ.alpha_activ, dim=0)
        prob_activ = prob_activ.detach().cpu().numpy()
        best_activ = prob_activ.argmax()
        mix_abit = 0
        abits = self.mix_activ.bits
        for i in range(len(abits)):
            mix_abit += prob_activ[i] * abits[i]
        prob_weight = F.softmax(self.mix_weight.alpha_weight, dim=0)
        prob_weight = prob_weight.detach().cpu().numpy()
        best_weight = prob_weight.argmax()
        mix_wbit = 0
        wbits = self.mix_weight.bits
        for i in range(len(wbits)):
            mix_wbit += prob_weight[i] * wbits[i]
        weight_shape = list(self.mix_weight.linear.weight.shape)
        print('idx {} with shape {}, activ alpha: {}, comp: {:.3f}M * {:.3f} * {:.3f}, '
              'memory: {:.3f}K * {:.3f}'.format(layer_idx, weight_shape, prob_activ, size_product,
                                                mix_abit, mix_wbit, memory_size, mix_abit))
        print('idx {} with shape {}, weight alpha: {}, comp: {:.3f}M * {:.3f} * {:.3f}, '
              'param: {:.3f}M * {:.3f}'.format(layer_idx, weight_shape, prob_weight, size_product,
                                               mix_abit, mix_wbit, self.param_size, mix_wbit))
        best_arch = {'best_activ': [best_activ], 'best_weight': [best_weight]}
        bitops = size_product * abits[best_activ] * wbits[best_weight]
        bita = memory_size * abits[best_activ]
        bitw = self.param_size * wbits[best_weight]
        dsps = size_product / dsp_factors_k11[wbits[best_weight]-2][abits[best_activ]-2]
        mixbitops = size_product * mix_abit * mix_wbit
        mixbita = memory_size * mix_abit
        mixbitw = self.param_size * mix_wbit
        mixdsps = 0
        for i in range(len(wbits)):
            for j in range(len(abits)):
                mixdsps += prob_weight[i] * prob_activ[j] / dsp_factors[wbits[i]-2][abits[j]-2]
        mixdsps *= size_product
        return best_arch, bitops, bita, bitw, mixbitops, mixbita, mixbitw, dsps, mixdsps
