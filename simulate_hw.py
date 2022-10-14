import argparse
import torch
import torch.nn.functional as F
import numpy as np

from export_hls import ConvParam
from mymodel import YOLOLayer
from test import get_prebox, hyp, bbox_iou
from torch.utils.data import DataLoader
from utils.datasets import LoadImagesAndLabels

class QConvLayer:
    def __init__(self, conv_param):
        self.conv = conv_param
        self.w = torch.tensor(self.conv.w, dtype = torch.int64)
    
    def __call__(self, x):
        if self.conv.icol < x.shape[-1]: # maxpool
            assert self.conv.irow*2, self.conv.icol*2 == x.shape[2:]
            x = F.max_pool2d(x.float(), kernel_size = 2, stride = 2).to(dtype=torch.int64)
        #print('convi', self.conv.n, x[0,0,:,0])

        x = F.conv2d(x, self.w, bias=None, stride=self.conv.s, padding=self.conv.p) # [N, OCH, OROW, OCOL]
        #print('convo', self.conv.n, x[0,0,:,0])
        och = x.shape[1]
        if self.conv.inc is not None:
            inc_ch = self.conv.inc.reshape((1, och, 1, 1))
            x *= inc_ch

        if hasattr(self.conv, 'bias'):
            bias_ch = self.conv.bias.reshape((1, och, 1, 1))
            x += bias_ch
        
        if hasattr(self.conv, 'lshift'):
            x >>= self.conv.lshift_T
        
        if hasattr(self.conv, 'obit'):
            x.clip_(0, 2**(self.conv.obit)-1)

        return x

class HWModel:
    def __init__(self, model_param):
        self.layers = [QConvLayer(conv_param) for conv_param in model_param]
        self.yololayer = YOLOLayer([[20,20], [20,20], [20,20], [20,20], [20,20], [20,20]])
        self.yololayer.eval()

    def __call__(self, x):
        assert len(x.shape) == 4 and x.dtype == torch.int64
        img_size = x.shape[-2:]

        if self.layers[0].conv.abit<8: # ImageInputQ
            x=x>>(8-self.layers[0].conv.abit) 

        for i, layer in enumerate(self.layers):
            x = layer(x)
        
        x = x.float() / self.layers[-1].conv.div

        io, p = self.yololayer(x, img_size)
        return io

def testdataset(hwmodel):
    batch_size = 1
    img_size = 320
    dataset = LoadImagesAndLabels(opt.datapath, img_size, opt.batch_size, rect=False, cache_labels=True, hyp=hyp, augment=False)
    batch_size = min(batch_size, len(dataset))
    dataloader = DataLoader(dataset,
                            batch_size=opt.batch_size,
                            #num_workers=min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8]),
                            pin_memory=True,
                            collate_fn=dataset.collate_fn)
    
    iou_sum = 0.0
    test_n = 0
    for imgs, targets, paths, shapes in dataloader:
        bn, _, height, width = imgs.shape  # batch size, channels, height, width
        test_n += bn

        imgs = imgs.to(dtype = torch.int64)
        inf_out = hwmodel(imgs)
        pre_box = get_prebox(inf_out)

        tbox = targets[..., 2:6] * torch.Tensor([width, height, width, height])
        ious = bbox_iou(pre_box, tbox)
        iou_sum += ious.sum()

        np.set_printoptions(precision = 2)
        for p in range(len(imgs)):
            print('pbox_xywh', pre_box[p].numpy(), 'tbox_xywh', tbox[p].numpy(), 'iou %.4f'%ious[p].item())
    
        meaniou = iou_sum / test_n
        # break

    print('iou', meaniou)

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('foldername', help='folder name in ./hls/, which contians model_param.pkl')
    parser.add_argument('--datapath', default='../dacsdc_dataset', help = 'test dataset path')
    parser.add_argument('-bs', '--batch-size', type=int, default=1, help = 'batch-size')
    opt = parser.parse_args()
    print(opt)
    
    x = torch.zeros([1,3,320,160], dtype=torch.int64)
    hwmodel = HWModel(torch.load('hls/'+opt.foldername+'/model_param.pkl'))
    
    testdataset(hwmodel)
