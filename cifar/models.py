import torch
import torch.nn as nn
import torch.nn.functional as F
from anypacking import quant_module as qm

class InputFactor:
    def __call__(self, pic):
        return pic * 255.0 / 256.0

class LeNet(nn.Module):
    def __init__(self):
        super(LeNet,self).__init__()
        conv=nn.Conv2d
        self.conv1 = conv(3,6,5)
        self.conv2 = conv(6,16,5)
        self.fc1 = nn.Linear(16*5*5,120)
        self.fc2 = nn.Linear(120,84)
        self.fc3 = nn.Linear(84,10)

    def forward(self,x):
        x = F.max_pool2d(F.relu(self.conv1(x)),(2,2))
        x = F.max_pool2d(F.relu(self.conv2(x)),2)
        x = x.view(x.size()[0],-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class VGG_small(nn.Module):
    def __init__(self, num_classes=10):
        super(VGG_small, self).__init__()
        self.pooling = nn.MaxPool2d(kernel_size=2, stride=2)
        self.nonlinear = nn.ReLU(inplace=True)

        self.layers = nn.Sequential(
            nn.Conv2d(3, 128, kernel_size=3, padding=1, bias=False), # 0
            nn.BatchNorm2d(128),
            self.nonlinear,

            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False), # 1
            self.pooling,
            nn.BatchNorm2d(128),
            self.nonlinear,

            nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=False), # 2
            nn.BatchNorm2d(256),
            self.nonlinear,

            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False), # 3
            self.pooling,
            nn.BatchNorm2d(256),
            self.nonlinear,

            nn.Conv2d(256, 512, kernel_size=3, padding=1, bias=False), # 4
            nn.BatchNorm2d(512),
            self.nonlinear,

            nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=False), # 5
            self.pooling,
            nn.BatchNorm2d(512),
            self.nonlinear,

            nn.Flatten(),
            nn.Linear(512*4*4, num_classes)
        )


    def forward(self, x):
        return self.layers(x)

class VGG_tiny(nn.Module):
    def __init__(self, num_classes=10):
        super(VGG_tiny, self).__init__()
        self.pooling = nn.MaxPool2d(kernel_size=2, stride=2)
        self.nonlinear = nn.ReLU(inplace=True)

        self.layers = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=False), # 0
            nn.BatchNorm2d(64),
            self.nonlinear,

            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False), # 1
            self.pooling,
            nn.BatchNorm2d(64),
            self.nonlinear,

            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False), # 2
            nn.BatchNorm2d(128),
            self.nonlinear,

            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False), # 3
            self.pooling,
            nn.BatchNorm2d(128),
            self.nonlinear,

            nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=False), # 4
            nn.BatchNorm2d(256),
            self.nonlinear,

            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False), # 5
            self.pooling,
            nn.BatchNorm2d(256),
            self.nonlinear,

            nn.Flatten(),
            nn.Linear(256*4*4, num_classes)
        )

    def forward(self, x):
        return self.layers(x)


class VGG_tiny_MixQ(nn.Module):
    def __init__(self, num_classes=10, share_weight = True):
        super(VGG_tiny_MixQ, self).__init__()
        self.pooling = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv_func = qm.MixActivConv2d
        conv_func = self.conv_func

        conv_kwargs = {'kernel_size':3, 'stride':1, 'padding':1, 'bias':False}
        qspace = {'wbits':[2,3,4,5,6,7,8], 'abits':[2,3,4,5,6,7,8], 'share_weight': share_weight}

        self.layers = nn.Sequential(
            conv_func(3, 64, ActQ = qm.ImageInputQ, **conv_kwargs, **qspace), # 0
            nn.BatchNorm2d(64),

            conv_func(64, 64, **conv_kwargs, **qspace), # 1
            self.pooling,
            nn.BatchNorm2d(64),

            conv_func(64, 128, **conv_kwargs, **qspace), # 2
            nn.BatchNorm2d(128),

            conv_func(128, 128, **conv_kwargs, **qspace), # 3
            self.pooling,
            nn.BatchNorm2d(128),

            conv_func(128, 256, **conv_kwargs, **qspace), # 4
            nn.BatchNorm2d(256),

            conv_func(256, 256, **conv_kwargs, **qspace), # 5
            self.pooling,
            nn.BatchNorm2d(256),

            nn.Flatten(),
            qm.QuantActivLinear(256*4*4, num_classes, bias=True, wbit=8, abit=8)
        )

    def forward(self, x):
        return self.layers(x)

    def fetch_best_arch(self):
        sum_bitops, sum_bita, sum_bitw, sum_dsps = 0, 0, 0, 0
        sum_mixbitops, sum_mixbita, sum_mixbitw, sum_mixdsps = 0, 0, 0, 0
        layer_idx = 0
        best_arch = None
        for m in self.modules():
            if isinstance(m, self.conv_func):
                layer_arch, bitops, bita, bitw, mixbitops, mixbita, mixbitw, dsps, mixdsps = m.fetch_best_arch(layer_idx)
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
                layer_idx += 1
        return best_arch, sum_bitops, sum_bita, sum_bitw, sum_mixbitops, sum_mixbita, sum_mixbitw, sum_dsps, sum_mixdsps

    def complexity_loss(self):
        size_product = []
        loss = 0
        for m in self.modules():
            if isinstance(m, self.conv_func):
                loss += m.complexity_loss()
                size_product += [m.size_product]
        normalizer = size_product[0].item()
        loss /= normalizer
        return loss

class VGG_tiny_FixQ(nn.Module):
    def __init__(self, num_classes=10, bitw = '444444', bita = '844444'):
        super(VGG_tiny_FixQ, self).__init__()
        self.pooling = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv_func = qm.QuantActivConv2d
        conv_func = self.conv_func

        assert(len(bitw)==0 or len(bitw)==6)
        assert(len(bita)==0 or len(bita)==6)
        if isinstance(bitw, str):
            bitw=list(map(int, bitw))
        if isinstance(bita, str):
            bita=list(map(int, bita))

        self.bitw = bitw
        self.bita = bita
        self.model_params = {'bitw': bitw, 'bita': bita}

        conv_kwargs = {'kernel_size':3, 'stride':1, 'padding':1, 'bias':False}

        self.layers = nn.Sequential(
            conv_func(3, 64, ActQ = qm.ImageInputQ, **conv_kwargs, wbit=bitw[0], abit=bita[0]), # 0
            nn.BatchNorm2d(64),

            conv_func(64, 64, **conv_kwargs, wbit=bitw[1], abit=bita[1]), # 1
            self.pooling,
            nn.BatchNorm2d(64),

            conv_func(64, 128, **conv_kwargs, wbit=bitw[2], abit=bita[2]), # 2
            nn.BatchNorm2d(128),

            conv_func(128, 128, **conv_kwargs, wbit=bitw[3], abit=bita[3]), # 3
            self.pooling,
            nn.BatchNorm2d(128),

            conv_func(128, 256, **conv_kwargs, wbit=bitw[4], abit=bita[4]), # 4
            nn.BatchNorm2d(256),

            conv_func(256, 256, **conv_kwargs, wbit=bitw[5], abit=bita[5]), # 5
            self.pooling,
            nn.BatchNorm2d(256),

            nn.Flatten(),
            qm.QuantActivLinear(256*4*4, num_classes, bias=True, wbit=8, abit=8)
        )

    def forward(self, x):
        return self.layers(x)

    def fetch_arch_info(self):
        sum_bitops, sum_bita, sum_bitw, sum_dsps = 0, 0, 0, 0
        layer_idx = 0
        for m in self.modules():
            if isinstance(m, self.conv_func):
                size_product = m.size_product.item()
                memory_size = m.memory_size.item()
                bitops = size_product * m.abit * m.wbit
                bita = m.memory_size.item() * m.abit
                bitw = m.param_size * m.wbit
                dsps = size_product / qm.dsp_factors[m.wbit-2][m.abit-2]
                weight_shape = list(m.conv.weight.shape)
                print('idx {} with shape {}, bitops: {:.3f}M * {} * {}, memory: {:.3f}K * {}, '
                      'param: {:.3f}M * {}, dsps: {:.3f}M'.format(layer_idx, weight_shape, size_product, m.abit,
                                                   m.wbit, memory_size, m.abit, m.param_size, m.wbit, dsps))
                sum_bitops += bitops
                sum_bita += bita
                sum_bitw += bitw
                sum_dsps += dsps
                layer_idx += 1
        return sum_bitops, sum_bita, sum_bitw, sum_dsps
