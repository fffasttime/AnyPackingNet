import os, sys
from typing import Dict
import argparse
import torch
import glob
import os

def select_weight_file():
    files = glob.glob('weights/*.pt')
    for i, s in enumerate(files):
        print('', i, s)
    sel = int(input('Select one .pt file (0-%d): '%(len(files)-1)))
    return os.path.split(files[sel])[-1][:-3]

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-w', '--weight', type=str, default=None, help='weights path')
    opt = parser.parse_args()
    if opt.weight is None: opt.weight = select_weight_file()

    model: Dict = torch.load('weights/' + opt.weight + '.pt', map_location='cpu')
    res = model['training_results']
    print(res)
    
    if 'model_params' in model:
        print(model['model_params'])

    if 'extra' in model:
        print(model['extra'])
