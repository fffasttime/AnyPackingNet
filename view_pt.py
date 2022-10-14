import os, sys
from typing import Dict
import torch

if __name__=='__main__':
    model: Dict = torch.load(sys.argv[1], map_location='cpu')
    res = model['training_results']
    print(res)
    
    if 'model_params' in model:
        print(model['model_params'])

    if 'extra' in model:
        print(model['extra'])
