import argparse
import os
import time
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import numpy as np
from tqdm import tqdm

import sys
sys.path.append('..')

from localconfig import data_path
import models
from test_acc import test
from utils import torch_utils
 
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    models.InputFactor(),
])
 
trainset = torchvision.datasets.CIFAR10(root=data_path, train=True,
                                        download=False, transform=transform_train)
classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


def train():
    torch_utils.init_seeds()

    model = models.VGG_tiny_FixQ(bitw = opt.bitw, bita = opt.bita)
    model.to(device)
    if opt.weights is not None:
        weights_file = 'weights/' + opt.weights + '.pt'
        chkpt = torch.load(weights_file, map_location=device)
        chkpt['model'] = {k: v for k, v in chkpt['model'].items() if 
                    model.state_dict()[k].numel() == v.numel()}
        model.load_state_dict(chkpt['model'], strict=False)

    results_file = 'results/%s.txt'%opt.name
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=opt.lr, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=opt.epochs, eta_min=opt.lr*0.01)

    model.train()

    start_epoch, epochs = 0, opt.epochs
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=opt.batch_size, shuffle=True, num_workers=2)
    test_best_acc = 0.0

    test(model, device)
    bops, bita, bitw, dsps = model.fetch_arch_info()
    print('model with bops: {:.3f}M, bita: {:.3f}K, bitw: {:.3f}M, dsps: {:.3f}M'.format(bops, bita, bitw, dsps))
    
    for epoch in range(start_epoch, epochs):
        model.train()
        mloss = macc = 0.
        pbar = tqdm(enumerate(train_loader), total=len(train_loader))
        for i, (inputs, labels) in pbar:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            correct = (predicted == labels).sum().item()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            mloss = (mloss*i + loss.item()) / (i+1)
            macc = (macc*i + correct/opt.batch_size) / (i+1)
            s = '%10s%10.2f%10.3g'%('%d/%d'%(epoch,epochs-1), macc*100, mloss)
            pbar.set_description(s)
        
        scheduler.step()
        results = test(model, device)
        with open(results_file, 'a') as f:
            f.write(s + '%10.2f%10.3g'% results + '\n')
        test_acc = results[0]
        test_best_acc = max(test_best_acc, test_acc)

        final_epoch = epoch == epochs-1
        if True or final_epoch:
            with open(results_file, 'r') as f:
                chkpt = {'epoch': epoch,
                            'training_results': f.read(),
                            'model': model.module.state_dict() if type(
                                model) is nn.parallel.DistributedDataParallel else model.state_dict(),
                            'optimizer': None if final_epoch else optimizer.state_dict(),
                            'model_params': model.model_params, # arch param
                            'extra': {'time': time.ctime(), 'name': opt.name}}
            # Save last checkpoint
            torch.save(chkpt, wdir + '%s_last.pt'%opt.name)
            
            if test_acc == test_best_acc:
                torch.save(chkpt, wdir + '%s_best.pt'%opt.name)
    
    print('Finished Training')

    # torch.save(net.state_dict(), 'lenet_cifar10.pth')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--name', default='VGG_tiny_FixQ', help='result and weight file name')
    parser.add_argument('-w', '--weights', default=None, help='weights path')
    parser.add_argument('-e', '--epochs', type=int, default=40) 
    parser.add_argument('--batch-size', type=int, default=128) 
    parser.add_argument('--device', default='', help='device id (i.e. 0 or 0,1 or cpu)')
    parser.add_argument('--lr', type=float, default=0.03)
    parser.add_argument('--bitw', type=str, default='')
    parser.add_argument('--bita', type=str, default='')

    opt = parser.parse_args()
    print(opt)
    wdir = 'weights' + os.sep  # weights dir
    last = wdir + '%s_last.pt'%opt.name

    device = torch_utils.select_device(opt.device, batch_size=opt.batch_size)

    train()
