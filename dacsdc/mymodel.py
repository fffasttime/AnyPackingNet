from typing import Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from quant_dorefa import conv2d_Q_fn, activation_quantize_fn
import anypacking.quant_module as qm

def create_grids(self, img_size=416, ng=(13, 13), device='cpu', type=torch.float32):
    nx, ny = ng  # x and y grid size
    self.img_size = max(img_size)
    self.stride = self.img_size / max(ng)

    # build xy offsets
    yv, xv = torch.meshgrid([torch.arange(ny), torch.arange(nx)])
    self.grid_xy = torch.stack((xv, yv), 2).to(device).type(type).view((1, 1, ny, nx, 2))

    # build wh gains
    self.anchor_vec = self.anchors.to(device) / self.stride
    self.anchor_wh = self.anchor_vec.view(1, self.na, 1, 1, 2).to(device).type(type)
    self.ng = torch.Tensor(ng).to(device)
    self.nx = nx
    self.ny = ny

class YOLOLayer(nn.Module):
    def __init__(self, anchors):
        super(YOLOLayer, self).__init__()

        self.anchors = torch.Tensor(anchors)
        self.na = len(anchors)  # number of anchors (3)
        self.no = 6  # number of outputs
        self.nx = 0  # initialize number of x gridpoints
        self.ny = 0  # initialize number of y gridpoints
    def forward(self, p, img_size):
        
        bs, _, ny, nx = p.shape  # bs, 255, 13, 13
        if (self.nx, self.ny) != (nx, ny):
            create_grids(self, img_size, (nx, ny), p.device, p.dtype)

        # p.view(bs, 255, 13, 13) -- > (bs, 3, 13, 13, 85)  # (bs, anchors, grid, grid, classes + xywh)
        p = p.view(bs, self.na, self.no, self.ny, self.nx).permute(0, 1, 3, 4, 2).contiguous()  # prediction

        if self.training:
            return p

        else:  # inference
            # s = 1.5  # scale_xy  (pxy = pxy * s - (s - 1) / 2)
            io = p.clone()  # inference output
            io[..., :2] = torch.sigmoid(io[..., :2]) + self.grid_xy  # xy
            io[..., 2:4] = torch.exp(io[..., 2:4]) * self.anchor_wh  # wh yolo method
            # io[..., 2:4] = ((torch.sigmoid(io[..., 2:4]) * 2) ** 3) * self.anchor_wh  # wh power method
            io[..., :4] *= self.stride  # 原始像素尺度

            
            torch.sigmoid_(io[..., 4:])
            
            
            return io.view(bs, -1, self.no), p

def fixq_fetch_arch_info(self):
    sum_bitops, sum_bita, sum_bitw, sum_dsps, sum_bram = 0, 0, 0, 0, 0
    layer_idx = 0
    for m in self.modules():
        if isinstance(m, self.conv_func):
            if m.wbit == 32 or m.abit == 32:
                layer_idx += 1
                continue

            size_product = m.size_product.item()
            memory_size = m.memory_size.item()
            bitops = size_product * m.abit * m.wbit
            bita = m.memory_size.item() * m.abit
            bitw = m.param_size * m.wbit
            if m.kernel_size == 1:
                dsp_factors = qm.dsp_factors_k11
            elif m.kernel_size == 3:
                dsp_factors = qm.dsp_factors_k33
            elif m.kernel_size == 5:
                dsp_factors = qm.dsp_factors_k55
            else:
                raise NotImplementedError
            
            if m.kernel_size == 1:
                bram_sw = 2 * m.in_width.item() * m.inplane
            else:
                bram_sw = (m.kernel_size+1)*m.in_width.item()*m.inplane
            bram_sw *= 1e-3

            dsps = size_product / dsp_factors[m.wbit-2][m.abit-2]
            weight_shape = list(m.conv.weight.shape)
            print('idx {} with shape {}, bitops: {:.3f}M * {} * {}, memory: {:.3f}K * {}, '
                    'param: {:.3f}M * {}, dsps: {:.3f}M, bram(wa|waf):({:.2f},{:.2f}|{:.1f},{:.1f},{:.1f})K'.format(layer_idx, weight_shape, size_product, m.abit,
                                                m.wbit, memory_size, m.abit, m.param_size, m.wbit, dsps,
                                                m.param_size * 1e3, bram_sw, bitw*1e3, bram_sw*m.abit, bitw*1e3 + bram_sw*m.abit))
            sum_bitops += bitops
            sum_bita += bita
            sum_bitw += bitw
            sum_dsps += dsps
            sum_bram += bitw*1e3 + bram_sw*m.abit
            layer_idx += 1
    return sum_bitops, sum_bita, sum_bitw, sum_dsps, sum_bram

def mixq_fetch_best_arch(self):
    sum_bitops, sum_bita, sum_bitw, sum_dsps = 0, 0, 0, 0
    sum_mixbitops, sum_mixbita, sum_mixbitw, sum_mixdsps = 0, 0, 0, 0
    sum_mixbram_weight, sum_mixbram_cache = 0, 0
    layer_idx = 0
    best_arch = None
    for m in self.modules():
        if isinstance(m, self.conv_func):
            layer_arch, bitops, bita, bitw, mixbitops, mixbita, mixbitw, dsps, mixdsps, mixbram_weight, mixbram_cache = m.fetch_best_arch(layer_idx)
            if best_arch is None:
                best_arch = layer_arch
            else:
                for key in layer_arch.keys():
                    if key not in best_arch:
                        best_arch[key] = layer_arch[key]
                    else:
                        best_arch[key].append(layer_arch[key][0])
            sum_bitops += bitops
            sum_bita += bita
            sum_bitw += bitw
            sum_mixbitops += mixbitops
            sum_mixbita += mixbita
            sum_mixbitw += mixbitw
            sum_dsps += dsps
            sum_mixdsps += mixdsps
            sum_mixbram_weight += mixbram_weight
            sum_mixbram_cache += mixbram_cache
            layer_idx += 1
    return best_arch, sum_bitops, sum_bita, sum_bitw, sum_mixbitops, sum_mixbita, sum_mixbitw, sum_dsps, sum_mixdsps, sum_mixbram_weight, sum_mixbram_cache

def mixq_complexity_loss(self):
    size_product = []
    loss = 0
    for m in self.modules():
        if isinstance(m, self.conv_func):
            loss += m.complexity_loss()
            size_product += [m.size_product]
    normalizer = size_product[0].item()
    loss /= normalizer
    return loss

def mixq_bram_loss(self):
    memory_sizes = []
    loss = 0
    for m in self.modules():
        if isinstance(m, self.conv_func):
            loss += m.bram_loss()
            memory_sizes += [m.memory_size.item()]
    normalizer = memory_sizes[0]
    loss /= normalizer
    return loss

class UltraNet_ismart(nn.Module):
    def __init__(self):
        super(UltraNet_ismart, self).__init__()
        W_BIT = 4
        A_BIT = 4
        conv2d_q = conv2d_Q_fn(W_BIT)
        conv2d_8 = conv2d_Q_fn(8)
        # act_q = activation_quantize_fn(4)

        self.layers = nn.Sequential(
            conv2d_8(3, 16, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(16),
            activation_quantize_fn(A_BIT),
            nn.MaxPool2d(2, stride=2),

            conv2d_q(16, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            activation_quantize_fn(A_BIT),

            nn.MaxPool2d(2, stride=2),

            conv2d_q(32, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            activation_quantize_fn(A_BIT),

            nn.MaxPool2d(2, stride=2),

            conv2d_q(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            activation_quantize_fn(A_BIT),

            nn.MaxPool2d(2, stride=2),

            conv2d_q(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            activation_quantize_fn(A_BIT),

            conv2d_q(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            activation_quantize_fn(A_BIT),

            conv2d_q(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            activation_quantize_fn(A_BIT),


            conv2d_q(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            activation_quantize_fn(A_BIT),


            # nn.Conv2d(256, 18, kernel_size=1, stride=1, padding=0)
            conv2d_8(64, 36, kernel_size=1, stride=1, padding=0)
            
        )
        self.yololayer = YOLOLayer([[20,20], [20,20], [20,20], [20,20], [20,20], [20,20]])
        self.yolo_layers = [self.yololayer]

    def forward(self, x):
        img_size = x.shape[-2:]
        yolo_out, out = [], []

        x = self.layers(x)
        x = self.yololayer(x, img_size)

        yolo_out.append(x)

        if self.training:  # train
            return yolo_out
        else:  # test
            io, p = zip(*yolo_out)  # inference output, training output
            return torch.cat(io, 1), p
        return x 


class UltraNet(nn.Module):
    def __init__(self):
        super(UltraNet, self).__init__()
        W_BIT = 4
        A_BIT = 4
        conv2d_q = conv2d_Q_fn(W_BIT)

        self.layers = nn.Sequential(
            conv2d_q(3, 16, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(16),
            activation_quantize_fn(A_BIT),
            nn.MaxPool2d(2, stride=2),

            conv2d_q(16, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            activation_quantize_fn(A_BIT),

            nn.MaxPool2d(2, stride=2),

            conv2d_q(32, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            activation_quantize_fn(A_BIT),

            nn.MaxPool2d(2, stride=2),

            conv2d_q(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            activation_quantize_fn(A_BIT),

            nn.MaxPool2d(2, stride=2),

            conv2d_q(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            activation_quantize_fn(A_BIT),

            conv2d_q(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            activation_quantize_fn(A_BIT),

            conv2d_q(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            activation_quantize_fn(A_BIT),


            conv2d_q(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            activation_quantize_fn(A_BIT),


            # nn.Conv2d(256, 18, kernel_size=1, stride=1, padding=0)
            conv2d_q(64, 36, kernel_size=1, stride=1, padding=0)
            
        )
        self.yololayer = YOLOLayer([[20,20], [20,20], [20,20], [20,20], [20,20], [20,20]])
        self.yolo_layers = [self.yololayer]

    def forward(self, x):
        img_size = x.shape[-2:]
        yolo_out, out = [], []

        x = self.layers(x)
        x = self.yololayer(x, img_size)

        yolo_out.append(x)

        if self.training:  # train
            return yolo_out
        else:  # test
            io, p = zip(*yolo_out)  # inference output, training output
            return torch.cat(io, 1), p
        return x 

class UltraNet_MixQ(nn.Module):
    def __init__(self, share_weight = False):
        super(UltraNet_MixQ, self).__init__()
        self.conv_func = qm.MixActivConv2d
        conv_func = self.conv_func

        conv_kwargs = {'kernel_size':3, 'stride':1, 'padding':1, 'bias':False}
        qspace = {'wbits':[2,3,4,5,6,7,8], 'abits':[2,3,4,5,6,7,8], 'share_weight':share_weight}

        self.layers = nn.Sequential(
            conv_func(3, 16, ActQ = qm.ImageInputQ, **conv_kwargs, **qspace), #0
            nn.BatchNorm2d(16),
            nn.MaxPool2d(2, stride=2),

            conv_func(16, 32, **conv_kwargs, **qspace),#1
            nn.BatchNorm2d(32),

            nn.MaxPool2d(2, stride=2),

            conv_func(32, 64, **conv_kwargs, **qspace),#2
            nn.BatchNorm2d(64),

            nn.MaxPool2d(2, stride=2),

            conv_func(64, 64, **conv_kwargs, **qspace),#3
            nn.BatchNorm2d(64),

            nn.MaxPool2d(2, stride=2),

            conv_func(64, 64, **conv_kwargs, **qspace),#4
            nn.BatchNorm2d(64),

            conv_func(64, 64, **conv_kwargs, **qspace),#5
            nn.BatchNorm2d(64),

            conv_func(64, 64, **conv_kwargs, **qspace),#6
            nn.BatchNorm2d(64),


            conv_func(64, 64, **conv_kwargs, **qspace),#7
            nn.BatchNorm2d(64),

            conv_func(64, 36, kernel_size = 1, stride =1, padding = 0, **qspace)#8
            
        )
        self.yololayer = YOLOLayer([[20,20], [20,20], [20,20], [20,20], [20,20], [20,20]])
        self.yolo_layers = [self.yololayer]

    def forward(self, x):
        img_size = x.shape[-2:]
        yolo_out, out = [], []
        
        x = self.layers(x)
        x = self.yololayer(x, img_size)

        yolo_out.append(x)

        if self.training:  # train
            return yolo_out
        else:  # test
            io, p = zip(*yolo_out)  # inference output, training output
            return torch.cat(io, 1), p
        return x 

    # def fetch_best_arch(self):
    fetch_best_arch = mixq_fetch_best_arch

    # def complexity_loss(self):
    complexity_loss= mixq_complexity_loss
    bram_loss = mixq_bram_loss

class UltraNet_FixQ(nn.Module):
    def __init__(self, bitw = '444444444', bita = '444444444'):
        super(UltraNet_FixQ, self).__init__()
        self.conv_func = qm.QuantActivConv2d
        conv_func = self.conv_func
        
        assert(len(bitw)==0 or len(bitw)==9)
        assert(len(bita)==0 or len(bita)==9)
        if isinstance(bitw, str):
            bitw=list(map(int, bitw))
        if isinstance(bita, str):
            bita=list(map(int, bita))

        self.bitw = bitw
        self.bita = bita
        self.model_params = {'bitw': bitw, 'bita': bita}

        conv_kwargs = {'kernel_size':3, 'stride':1, 'padding':1, 'bias':False}

        self.layers = nn.Sequential(
            conv_func(3, 16, ActQ = qm.ImageInputQ, **conv_kwargs, wbit=bitw[0], abit=bita[0]),#0
            nn.BatchNorm2d(16),
            nn.MaxPool2d(2, stride=2),

            conv_func(16, 32, **conv_kwargs, wbit=bitw[1], abit=bita[1]),#1
            nn.BatchNorm2d(32),

            nn.MaxPool2d(2, stride=2),

            conv_func(32, 64, **conv_kwargs, wbit=bitw[2], abit=bita[2]),#2
            nn.BatchNorm2d(64),

            nn.MaxPool2d(2, stride=2),

            conv_func(64, 64, **conv_kwargs, wbit=bitw[3], abit=bita[3]),#3
            nn.BatchNorm2d(64),

            nn.MaxPool2d(2, stride=2),

            conv_func(64, 64, **conv_kwargs, wbit=bitw[4], abit=bita[4]),#4
            nn.BatchNorm2d(64),

            conv_func(64, 64, **conv_kwargs, wbit=bitw[5], abit=bita[5]),#5
            nn.BatchNorm2d(64),

            conv_func(64, 64, **conv_kwargs, wbit=bitw[6], abit=bita[6]),#6
            nn.BatchNorm2d(64),


            conv_func(64, 64, **conv_kwargs, wbit=bitw[7], abit=bita[7]),#7
            nn.BatchNorm2d(64),

            conv_func(64, 36, kernel_size = 1, stride =1, padding = 0, wbit=bitw[8], abit=bita[8])#8
            
        )
        self.yololayer = YOLOLayer([[20,20], [20,20], [20,20], [20,20], [20,20], [20,20]])
        self.yolo_layers = [self.yololayer]

    def forward(self, x):
        img_size = x.shape[-2:]
        
        x = self.layers(x)
        p = self.yololayer(x, img_size) # train: p; test: (io, p)

        if self.training:  # train
            return [p]
        else:  # test
            return p[0], (p[1],) # inference output, training output

    # def fetch_arch_info(self):
    fetch_arch_info = fixq_fetch_arch_info

class UltraNetFloat(nn.Module):
    def __init__(self):
        super(UltraNetFloat, self).__init__()

        self.layers = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2),

            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(2, stride=2),

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(2, stride=2),

            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(2, stride=2),

            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),


            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),


            # nn.Conv2d(256, 18, kernel_size=1, stride=1, padding=0)
            nn.Conv2d(64, 36, kernel_size=1, stride=1, padding=0)
            
        )
        self.yololayer = YOLOLayer([[20,20], [20,20], [20,20], [20,20], [20,20], [20,20]])
        self.yolo_layers = [self.yololayer]

    def forward(self, x):
        img_size = x.shape[-2:]
        yolo_out, out = [], []

        x = self.layers(x)
        x = self.yololayer(x, img_size)

        yolo_out.append(x)

        if self.training:  # train
            return yolo_out
        else:  # test
            io, p = zip(*yolo_out)  # inference output, training output
            return torch.cat(io, 1), p
        return x

class ReorgLayer(nn.Module):
    def __init__(self, stride=2):
        super(ReorgLayer, self).__init__()
        self.stride = stride
    def forward(self, x):
        stride = self.stride
        assert(x.data.dim() == 4)
        B = x.data.size(0)
        C = x.data.size(1)
        H = x.data.size(2)
        W = x.data.size(3)
        assert(H % stride == 0)
        assert(W % stride == 0)
        ws = stride
        hs = stride
        x = x.view([B, C, H//hs, hs, W//ws, ws]).transpose(3, 4).contiguous()
        x = x.view([B, C, H//hs*W//ws, hs*ws]).transpose(2, 3).contiguous()
        x = x.view([B, C, hs*ws, H//hs, W//ws]).transpose(1, 2).contiguous()
        x = x.view([B, hs*ws*C, H//hs, W//ws])
        return x

class UltraNetBypassFloat(nn.Module):
    def __init__(self):
        super(UltraNetBypassFloat, self).__init__()
        self.reorg = ReorgLayer(stride=2)

        self.layers_p1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2),

            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2),

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2)
        )

        self.layers_p2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        self.layers_p3 = nn.Sequential(
            nn.MaxPool2d(2, stride=2),

            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        self.layers_p4 = nn.Sequential(
            nn.Conv2d(320, 64, kernel_size=3, stride=1, padding=1, bias=False),   # cat p2--64→64*4 + 64
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 36, kernel_size=1, stride=1, padding=0)
        )
        self.yololayer = YOLOLayer([[20, 20], [20, 20], [20, 20], [20, 20], [20, 20], [20, 20]])
        self.yolo_layers = [self.yololayer]

    def forward(self, x):
        img_size = x.shape[-2:]
        yolo_out, out = [], []

        x_p1 = self.layers_p1(x)
        x_p2 = self.layers_p2(x_p1)
        x_p2_reorg = self.reorg(x_p2)
        x_p3 = self.layers_p3(x_p2)
        x_p4_in = torch.cat([x_p2_reorg, x_p3], 1)
        x_p4 = self.layers_p4(x_p4_in)

        x = self.yololayer(x_p4, img_size)

        yolo_out.append(x)

        if self.training:  # train
            return yolo_out
        else:  # test
            io, p = zip(*yolo_out)  # inference output, training output
            return torch.cat(io, 1), p
        return x

class UltraNetBypass_MixQ(nn.Module):
    def __init__(self, share_weight = False):
        super(UltraNetBypass_MixQ, self).__init__()
        self.reorg = ReorgLayer(stride=2)
        self.conv_func = qm.MixActivConv2d
        conv_func = self.conv_func

        conv_kwargs = {'kernel_size':3, 'stride':1, 'padding':1, 'bias':False}
        qspace = {'wbits':[2,3,4,5,6,7,8], 'abits':[2,3,4,5,6,7,8], 'share_weight':share_weight}

        self.layers_p1 = nn.Sequential(
            conv_func(3, 16, ActQ = qm.ImageInputQ, **conv_kwargs, **qspace),
            nn.BatchNorm2d(16),
            nn.MaxPool2d(2, stride=2),

            conv_func(16, 32, **conv_kwargs, **qspace),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2, stride=2),

            conv_func(32, 64, **conv_kwargs, **qspace),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2, stride=2)
        )

        self.layers_p2 = nn.Sequential(
            conv_func(64, 64, **conv_kwargs, **qspace),
            nn.BatchNorm2d(64),
        )

        self.layers_p3 = nn.Sequential(
            nn.MaxPool2d(2, stride=2),

            conv_func(64, 64, **conv_kwargs, **qspace),
            nn.BatchNorm2d(64),

            conv_func(64, 64, **conv_kwargs, **qspace),
            nn.BatchNorm2d(64),

            conv_func(64, 64, **conv_kwargs, **qspace),
            nn.BatchNorm2d(64),
        )

        self.layers_p4 = nn.Sequential(
            conv_func(320, 64, **conv_kwargs, **qspace),   # cat p2--64→64*4 + 64
            nn.BatchNorm2d(64),

            conv_func(64, 36, kernel_size = 1, stride =1, padding = 0, **qspace)
        )
        self.yololayer = YOLOLayer([[20, 20], [20, 20], [20, 20], [20, 20], [20, 20], [20, 20]])
        self.yolo_layers = [self.yololayer]

    def forward(self, x):
        img_size = x.shape[-2:]
        yolo_out, out = [], []

        x_p1 = self.layers_p1(x)
        x_p2 = self.layers_p2(x_p1)
        x_p2_reorg = self.reorg(x_p2)
        x_p3 = self.layers_p3(x_p2)
        x_p4_in = torch.cat([x_p2_reorg, x_p3], 1)
        x_p4 = self.layers_p4(x_p4_in)

        x = self.yololayer(x_p4, img_size)

        yolo_out.append(x)

        if self.training:  # train
            return yolo_out
        else:  # test
            io, p = zip(*yolo_out)  # inference output, training output
            return torch.cat(io, 1), p
        return x

    # def fetch_best_arch(self):
    fetch_best_arch = mixq_fetch_best_arch

    # def complexity_loss(self):
    complexity_loss= mixq_complexity_loss
    bram_loss = mixq_bram_loss

class UltraNetBypass_FixQ(nn.Module):
    def __init__(self, bitw = '444444444', bita = '444444444'):
        super(UltraNetBypass_FixQ, self).__init__()
        self.reorg = ReorgLayer(stride=2)
        self.conv_func = qm.QuantActivConv2d
        conv_func = self.conv_func
        
        assert(len(bitw)==0 or len(bitw)==9)
        assert(len(bita)==0 or len(bita)==9)
        if isinstance(bitw, str):
            bitw=list(map(int, bitw))
        if isinstance(bita, str):
            bita=list(map(int, bita))

        self.bitw = bitw
        self.bita = bita
        self.model_params = {'bitw': bitw, 'bita': bita}

        conv_kwargs = {'kernel_size':3, 'stride':1, 'padding':1, 'bias':False}

        self.layers_p1 = nn.Sequential(
            conv_func(3, 16, ActQ = qm.ImageInputQ, **conv_kwargs, wbit=bitw[0], abit=bita[0]),
            nn.BatchNorm2d(16),
            nn.MaxPool2d(2, stride=2),

            conv_func(16, 32, **conv_kwargs, wbit=bitw[1], abit=bita[1]),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2, stride=2),

            conv_func(32, 64, **conv_kwargs, wbit=bitw[2], abit=bita[2]),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2, stride=2)
        )

        self.layers_p2 = nn.Sequential(
            conv_func(64, 64, **conv_kwargs, wbit=bitw[3], abit=bita[3]),
            nn.BatchNorm2d(64),
        )

        self.layers_p3 = nn.Sequential(
            nn.MaxPool2d(2, stride=2),

            conv_func(64, 64, **conv_kwargs, wbit=bitw[4], abit=bita[4]),
            nn.BatchNorm2d(64),

            conv_func(64, 64, **conv_kwargs, wbit=bitw[5], abit=bita[5]),
            nn.BatchNorm2d(64),

            conv_func(64, 64, **conv_kwargs, wbit=bitw[6], abit=bita[6]),
            nn.BatchNorm2d(64),
        )

        self.layers_p4 = nn.Sequential(
            conv_func(320, 64, **conv_kwargs, wbit=bitw[7], abit=bita[7]),   # cat p2--64→64*4 + 64
            nn.BatchNorm2d(64),

            conv_func(64, 36, kernel_size = 1, stride =1, padding = 0, wbit=bitw[8], abit=bita[8])
        )
        self.yololayer = YOLOLayer([[20, 20], [20, 20], [20, 20], [20, 20], [20, 20], [20, 20]])
        self.yolo_layers = [self.yololayer]

    def forward(self, x):
        img_size = x.shape[-2:]
        yolo_out, out = [], []

        x_p1 = self.layers_p1(x)
        x_p2 = self.layers_p2(x_p1)
        x_p2_reorg = self.reorg(x_p2)
        x_p3 = self.layers_p3(x_p2)
        x_p4_in = torch.cat([x_p2_reorg, x_p3], 1)
        x_p4 = self.layers_p4(x_p4_in)

        x = self.yololayer(x_p4, img_size)

        yolo_out.append(x)

        if self.training:  # train
            return yolo_out
        else:  # test
            io, p = zip(*yolo_out)  # inference output, training output
            return torch.cat(io, 1), p
        return x

    # def fetch_arch_info(self):
    fetch_arch_info = fixq_fetch_arch_info

class SkyNetFloat(nn.Module):
    def __init__(self):
        super(SkyNetFloat, self).__init__()
    
        self.header = torch.IntTensor([0,0,0,0])
        self.seen = 0
        
        self.reorg = ReorgLayer(stride=2)
        def conv_dw(inp, oup, stride):
            return nn.Sequential(
                nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
                nn.BatchNorm2d(inp),
                nn.ReLU6(inplace=True),
                
                nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU6(inplace=True),
            )
        self.layers = nn.Sequential(
            conv_dw( 3,  48, 1),    #dw1
            nn.MaxPool2d(kernel_size=2, stride=2),
            conv_dw( 48,  96, 1),   #dw2
            nn.MaxPool2d(kernel_size=2, stride=2),
            conv_dw( 96, 192, 1),   #dw3
            nn.MaxPool2d(kernel_size=2, stride=2),
            conv_dw(192, 384, 1),   #dw4
            conv_dw(384, 512, 1),   #dw5
            conv_dw(512, 96, 1),
            nn.Conv2d(96, 36, 1, 1,bias=False),
        )
        self.yololayer = YOLOLayer([[20,20], [20,20], [20,20], [20,20], [20,20], [20,20]])

        self.yolo_layers = [self.yololayer]

    def forward(self, x):
        img_size = x.shape[-2:]
        yolo_out, out = [], []

        x = self.layers(x)
        x = self.yololayer(x, img_size)

        yolo_out.append(x)

        if self.training:  # train
            return yolo_out
        else:  # test
            io, p = zip(*yolo_out)  # inference output, training output
            return torch.cat(io, 1), p
        return x

class SkyNet_MixQ(nn.Module):
    def __init__(self, share_weight = False):
        super(SkyNet_MixQ, self).__init__()

        self.conv_func = qm.MixActivConv2d
        conv_func = self.conv_func

        qspace = {'wbits':[2,3,4,5,6,7,8], 'abits':[2,3,4,5,6,7,8], 'share_weight':share_weight}

        channels = [3,48,96,192,384,512,96]

        layers = []
        for i in range(6):
            inp, oup = channels[i], channels[i+1]
            layers.append(conv_func(
                inp, inp, kernel_size=3, stride=1, padding=1, bias=False, groups = inp,
                ActQ = qm.ImageInputQ if i==0 else qm.HWGQ, **qspace)) # dwconv
            layers.append(nn.BatchNorm2d(channels[i]))

            layers.append(conv_func(
                inp, oup, kernel_size=1, stride=1, padding=0, bias=False, 
                **qspace)) # pwconv

            layers.append(nn.BatchNorm2d(channels[i+1]))

            if i<3:
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))

        layers.append(conv_func(
                channels[-1], 36, kernel_size=1, stride=1, padding=0, bias=False, 
                **qspace))

        self.layers = nn.Sequential(*layers)

        self.yololayer = YOLOLayer([[20,20], [20,20], [20,20], [20,20], [20,20], [20,20]])
        self.yolo_layers = [self.yololayer]

    def forward(self, x):
        img_size = x.shape[-2:]
        
        x = self.layers(x)
        p = self.yololayer(x, img_size) # train: p; test: (io, p)

        if self.training:  # train
            return [p]
        else:  # test
            return p[0], (p[1],) # inference output, training output

    # def fetch_best_arch(self):
    fetch_best_arch = mixq_fetch_best_arch

    # def complexity_loss(self):
    complexity_loss= mixq_complexity_loss
    bram_loss = mixq_bram_loss

class SkyNetk5_MixQ(nn.Module):
    def __init__(self, share_weight = False):
        super(SkyNetk5_MixQ, self).__init__()

        self.conv_func = qm.MixActivConv2d
        conv_func = self.conv_func

        qspace = {'wbits':[2,3,4,5,6,7,8], 'abits':[2,3,4,5,6,7,8], 'share_weight':share_weight}

        channels = [3,48,96,192,384,512,96]

        layers = []
        for i in range(6):
            inp, oup = channels[i], channels[i+1]
            layers.append(conv_func(
                inp, inp, kernel_size=5, stride=1, padding=2, bias=False, groups = inp,
                ActQ = qm.ImageInputQ if i==0 else qm.HWGQ, **qspace)) # dwconv
            layers.append(nn.BatchNorm2d(channels[i]))

            layers.append(conv_func(
                inp, oup, kernel_size=1, stride=1, padding=0, bias=False, 
                **qspace)) # pwconv

            layers.append(nn.BatchNorm2d(channels[i+1]))

            if i<3:
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))

        layers.append(conv_func(
                channels[-1], 36, kernel_size=1, stride=1, padding=0, bias=False, 
                **qspace))

        self.layers = nn.Sequential(*layers)

        self.yololayer = YOLOLayer([[20,20], [20,20], [20,20], [20,20], [20,20], [20,20]])
        self.yolo_layers = [self.yololayer]

    def forward(self, x):
        img_size = x.shape[-2:]
        
        x = self.layers(x)
        p = self.yololayer(x, img_size) # train: p; test: (io, p)

        if self.training:  # train
            return [p]
        else:  # test
            return p[0], (p[1],) # inference output, training output

    # def fetch_best_arch(self):
    fetch_best_arch = mixq_fetch_best_arch

    # def complexity_loss(self):
    complexity_loss= mixq_complexity_loss
    bram_loss = mixq_bram_loss

class SkyNet_FixQ(nn.Module):
    def __init__(self, bitw='', bita=''):
        super(SkyNet_FixQ, self).__init__()
        self.conv_func = qm.QuantActivConv2d
        conv_func = self.conv_func
        
        assert(len(bitw)==0 or len(bitw)==13)
        assert(len(bita)==0 or len(bita)==13)
        if len(bitw)==0: bitw='5'*13
        if len(bita)==0: bita='8'*13
        if isinstance(bitw, str):
            bitw=list(map(int, bitw))
        if isinstance(bita, str):
            bita=list(map(int, bita))

        self.bitw = bitw
        self.bita = bita
        self.model_params = {'bitw': bitw, 'bita': bita}

        channels = [3,48,96,192,384,512,96]

        layers = []
        for i in range(6):
            inp, oup = channels[i], channels[i+1]
            layers.append(conv_func(
                inp, inp, kernel_size=3, stride=1, padding=1, bias=False, groups = inp,
                ActQ = qm.ImageInputQ if i==0 else qm.HWGQ, wbit=bitw[i*2], abit=bita[i*2])) # dwconv
            layers.append(nn.BatchNorm2d(channels[i]))

            layers.append(conv_func(
                inp, oup, kernel_size=1, stride=1, padding=0, bias=False, 
                wbit=bitw[i*2+1], abit=bita[i*2+1])) # pwconv

            layers.append(nn.BatchNorm2d(channels[i+1]))

            if i<3:
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))

        layers.append(conv_func(
                channels[-1], 36, kernel_size=1, stride=1, padding=0, bias=False, 
                wbit=bitw[-1], abit=bita[-1]))

        self.layers = nn.Sequential(*layers)

        self.yololayer = YOLOLayer([[20,20], [20,20], [20,20], [20,20], [20,20], [20,20]])
        self.yolo_layers = [self.yololayer]

    def forward(self, x):
        img_size = x.shape[-2:]
        
        x = self.layers(x)
        p = self.yololayer(x, img_size) # train: p; test: (io, p)

        if self.training:  # train
            return [p]
        else:  # test
            return p[0], (p[1],) # inference output, training output

    # def fetch_arch_info(self):
    fetch_arch_info = fixq_fetch_arch_info

class SkyNetk5_FixQ(nn.Module):
    def __init__(self, bitw = '5555555555555', bita = '8888888888888'):
        super(SkyNetk5_FixQ, self).__init__()
        self.conv_func = qm.QuantActivConv2d
        conv_func = self.conv_func
        
        assert(len(bitw)==0 or len(bitw)==13)
        assert(len(bita)==0 or len(bita)==13)
        if len(bitw)==0: bitw='5'*13
        if len(bita)==0: bita='8'*13
        if isinstance(bitw, str):
            bitw=list(map(int, bitw))
        if isinstance(bita, str):
            bita=list(map(int, bita))

        self.bitw = bitw
        self.bita = bita
        self.model_params = {'bitw': bitw, 'bita': bita}

        channels = [3,48,96,192,384,512,96]

        layers = []
        for i in range(6):
            inp, oup = channels[i], channels[i+1]
            layers.append(conv_func(
                inp, inp, kernel_size=5, stride=1, padding=2, bias=False, groups = inp,
                ActQ = qm.ImageInputQ if i==0 else qm.HWGQ, wbit=bitw[i*2], abit=bita[i*2])) # dwconv
            layers.append(nn.BatchNorm2d(channels[i]))

            layers.append(conv_func(
                inp, oup, kernel_size=1, stride=1, padding=0, bias=False, 
                wbit=bitw[i*2+1], abit=bita[i*2+1])) # pwconv

            layers.append(nn.BatchNorm2d(channels[i+1]))

            if i<3:
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))

        layers.append(conv_func(
                channels[-1], 36, kernel_size=1, stride=1, padding=0, bias=False, 
                wbit=bitw[-1], abit=bita[-1]))

        self.layers = nn.Sequential(*layers)

        self.yololayer = YOLOLayer([[20,20], [20,20], [20,20], [20,20], [20,20], [20,20]])
        self.yolo_layers = [self.yololayer]

    def forward(self, x):
        img_size = x.shape[-2:]
        
        x = self.layers(x)
        p = self.yololayer(x, img_size) # train: p; test: (io, p)

        if self.training:  # train
            return [p]
        else:  # test
            return p[0], (p[1],) # inference output, training output

    # def fetch_arch_info(self):
    fetch_arch_info = fixq_fetch_arch_info

class SkyNetk5Float(nn.Module):
    def __init__(self):
        super(SkyNetk5Float, self).__init__()
    
        self.header = torch.IntTensor([0,0,0,0])
        self.seen = 0
        
        self.reorg = ReorgLayer(stride=2)
        def conv_dw(inp, oup, stride):
            return nn.Sequential(
                nn.Conv2d(inp, inp, 5, stride, 2, groups=inp, bias=False),
                nn.BatchNorm2d(inp),
                nn.ReLU6(inplace=True),
                
                nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU6(inplace=True),
            )
        self.layers = nn.Sequential(
            conv_dw( 3,  48, 1),    #dw1
            nn.MaxPool2d(kernel_size=2, stride=2),
            conv_dw( 48,  96, 1),   #dw2
            nn.MaxPool2d(kernel_size=2, stride=2),
            conv_dw( 96, 192, 1),   #dw3
            nn.MaxPool2d(kernel_size=2, stride=2),
            conv_dw(192, 384, 1),   #dw4
            conv_dw(384, 512, 1),   #dw5
            conv_dw(512, 96, 1),
            nn.Conv2d(96, 36, 1, 1,bias=False),
        )
        self.yololayer = YOLOLayer([[20,20], [20,20], [20,20], [20,20], [20,20], [20,20]])

        self.yolo_layers = [self.yololayer]

    def forward(self, x):
        img_size = x.shape[-2:]
        yolo_out, out = [], []

        x = self.layers(x)
        x = self.yololayer(x, img_size)

        yolo_out.append(x)

        if self.training:  # train
            return yolo_out
        else:  # test
            io, p = zip(*yolo_out)  # inference output, training output
            return torch.cat(io, 1), p
        return x

class SkyNetBypassFloat(nn.Module):
    def __init__(self):
        super(SkyNetBypassFloat, self).__init__()
    
        self.header = torch.IntTensor([0,0,0,0])
        self.seen = 0
        
        self.reorg = ReorgLayer(stride=2)
        def conv_dw(inp, oup, stride):
            return nn.Sequential(
                nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
                nn.BatchNorm2d(inp),
                nn.ReLU6(inplace=True),
                
                nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU6(inplace=True),
            )
        self.model_p1 = nn.Sequential(
            conv_dw( 3,  48, 1),    #dw1
            nn.MaxPool2d(kernel_size=2, stride=2),
            conv_dw( 48,  96, 1),   #dw2
            nn.MaxPool2d(kernel_size=2, stride=2),
            conv_dw( 96, 192, 1),   #dw3
        )    
        self.model_p2 = nn.Sequential(    
            nn.MaxPool2d(kernel_size=2, stride=2),
            conv_dw(192, 384, 1),   #dw4
            conv_dw(384, 512, 1),   #dw5
        )
        self.model_p3 = nn.Sequential(  #cat dw3(ch:192 -> 768) and dw5(ch:512)
            conv_dw(1280, 96, 1),
            nn.Conv2d(96, 36, 1, 1,bias=False),
        )
        self.yololayer = YOLOLayer([[20,20], [20,20], [20,20], [20,20], [20,20], [20,20]])

        self.yolo_layers = [self.yololayer]
        # self._initialize_weights()

    def forward(self, x):
        img_size = x.shape[-2:]
        yolo_out, out = [], []

        x_p1 = self.model_p1(x)
        x_p1_reorg = self.reorg(x_p1)
        x_p2 = self.model_p2(x_p1)
        x_p3_in = torch.cat([x_p1_reorg, x_p2], 1)
        x = self.model_p3(x_p3_in)
        x = self.yololayer(x, img_size)

        yolo_out.append(x)

        if self.training:  # train
            return yolo_out
        else:  # test
            io, p = zip(*yolo_out)  # inference output, training output
            return torch.cat(io, 1), p
        return x