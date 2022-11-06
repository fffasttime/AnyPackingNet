import argparse
from typing import Dict
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm

import sys
sys.path.append('..')

import models
from localconfig import data_path
from utils.view_pt import select_weight_file
from utils import torch_utils

transform_test = transforms.Compose([
    transforms.ToTensor(),
])

testset = torchvision.datasets.CIFAR10(root=data_path, train=False,
                                       download=False, transform=transform_test)

def test(model, device, batch_size = 64, num_batch = -1):
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)
    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        mloss = macc = 0.
        pbar = tqdm(enumerate(testloader), total=len(testloader))
        for i, (inputs, labels) in pbar:
            if i == num_batch: break
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            correct = (predicted == labels).sum().item()
            
            loss = criterion(outputs, labels)
            mloss = (mloss*i + loss.item()) / (i+1)
            macc = (macc*i + correct/batch_size) / (i+1)
            s = ' '*10 + '%10.2f%10.3g'%(macc*100, mloss)
            pbar.set_description(s)
    
    return macc * 100, mloss

if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='test.py')
    parser.add_argument('-m', '--model', type=str, default='VGG_tiny_FixQ', help='model name')
    parser.add_argument('-w', '--weight', default=None, help='weights path')
    parser.add_argument('-bs', '--batch-size', type=int, default=64, help='size of each image batch')
    parser.add_argument('--device', default='', help='device id (i.e. 0 or 0,1) or cpu')
    parser.add_argument('--verbose', action='store_true', help = 'show predict value result')
    parser.add_argument('-nb', '--num-batch', type=int, default='-1', help='num of batchs to run, -1 for full dataset')
    opt = parser.parse_args()
    print(opt)
    if opt.weight is None: opt.weight = select_weight_file()
    
    device = torch_utils.select_device(opt.device, batch_size=opt.batch_size)
    ptfile: Dict = torch.load('weights/' + opt.weight+'.pt', map_location=device)
    model_params = ptfile.setdefault('model_params', {})
    model = getattr(models, opt.model)(**model_params).to(device)
    model.load_state_dict(ptfile['model'])

    # Test
    res = test(model, device, batch_size=opt.batch_size, num_batch=opt.num_batch)
    print(('%s %s.pt\nacc %.2f, loss %.4f')%(opt.model, opt.weight, *res))
