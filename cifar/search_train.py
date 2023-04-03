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
from utils import torch_utils
from test_acc import test
 
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

    model = models.VGG_tiny_MixQ(10, not opt.noshare)
    model.to(device)

    results_file = 'results/%s.txt'%opt.name
    
    criterion = nn.CrossEntropyLoss()
    
    params, alpha_params = [], []
    for name, param in model.named_parameters():
        if 'alpha' in name:
            alpha_params += [param]
        else:
            params += [param]
    optimizer = optim.SGD(params, lr=opt.lr, momentum=0.9, weight_decay=5e-4)
    arch_optimizer = torch.optim.SGD(alpha_params, opt.lra, momentum=0.9, weight_decay=5e-4)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=opt.epochs, eta_min=opt.lr*0.01) 
    arch_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            arch_optimizer, T_max=opt.epochs, eta_min=opt.lr*0.3)

    model.train()

    start_epoch, epochs = 0, opt.epochs
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=opt.batch_size, shuffle=True, num_workers=2)
    test_best_acc = 0.0

    for epoch in range(start_epoch, epochs):
        model.train()
        mloss = macc = 0.
        pbar = tqdm(enumerate(train_loader), total=len(train_loader))
        for i, (inputs, labels) in pbar:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            arch_optimizer.zero_grad()
            
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            correct = (predicted == labels).sum().item()
            loss = criterion(outputs, labels)
            
            if opt.complexity_decay != 0 or opt.complexity_decay_trivial!=0:
                loss_complexity = opt.complexity_decay * model.complexity_loss() + \
                                  opt.complexity_decay_trivial * model.complexity_loss_trivial()
                loss += loss_complexity

            loss.backward()
            optimizer.step()
            arch_optimizer.step()
            
            mloss = (mloss*i + loss.item()) / (i+1)
            macc = (macc*i + correct/opt.batch_size) / (i+1)
            s = '%10s%10.2f%10.3g'%('%d/%d'%(epoch,epochs-1), macc*100, mloss)
            pbar.set_description(s)

        print('========= architecture =========')
        best_arch, bitops, bita, bitw, mixbitops, mixbita, mixbitw, dsps, mixdsps  = model.fetch_best_arch()
        print('best model with bitops: {:.3f}M, bita: {:.3f}K, bitw: {:.3f}M, dsps: {:.3f}M'.format(
            bitops, bita, bitw, dsps))
        print('expected model with bitops: {:.3f}M, bita: {:.3f}K, bitw: {:.3f}M, dsps: {:.3f}M'.format(
            mixbitops, mixbita, mixbitw, mixdsps))
        bestw_str = "".join([str(x+2) for x in best_arch["best_weight"]])
        besta_str = "".join([str(x+2) for x in best_arch["best_activ"]])
        print(f'best_weight: {best_arch["best_weight"]}')
        print(f'best_activ: {best_arch["best_activ"]}')
        
        scheduler.step()
        arch_scheduler.step()

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
                            'arch_optimizer': None if final_epoch else arch_optimizer.state_dict(),
                            'extra': {'time': time.ctime(), 'name': opt.name, 'bestw': bestw_str, 'besta': besta_str}}
            # Save last checkpoint
            torch.save(chkpt, wdir + '%s_last.pt'%opt.name)
            
            if test_acc == test_best_acc:
                torch.save(chkpt, wdir + '%s_best.pt'%opt.name)
    
    print('Finished Training')

    with open('results.csv', 'a') as f:
        print("mixed,%s,%d/%d, , , , ,%.1f,%.1f, ,%s,%s,%d,%d,%.3f,%.3f"%
              (opt.name,epochs-1,epochs,macc*100,(test_acc+test_best_acc)/2,
               bestw_str,besta_str,
               int(round(bitops)), int(round(mixbitops)), dsps, mixdsps), file=f)

    # torch.save(net.state_dict(), 'lenet_cifar10.pth')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=40) 
    parser.add_argument('--batch-size', type=int, default=128) 
    parser.add_argument('--device', default='', help='device id (i.e. 0 or 0,1 or cpu)')
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--name', default='', help='result and weight file name')
    parser.add_argument('--noshare', action='store_true', help='no share weight')
    parser.add_argument('--complexity-decay', '--cd', default=0, type=float, metavar='W', help='complexity decay (default: 0)')
    parser.add_argument('--complexity-decay-trivial', '--cdt', default=0, type=float, metavar='W', help='complexity decay w/o hardware-aware')
    parser.add_argument('--lra', '--learning-rate-alpha', default=0.1, type=float, metavar='LR', help='initial alpha learning rate')

    opt = parser.parse_args()
    print(opt)
    wdir = 'weights' + os.sep  # weights dir
    last = wdir + '%s_last.pt'%opt.name

    device = torch_utils.select_device(opt.device, batch_size=opt.batch_size)

    train()
