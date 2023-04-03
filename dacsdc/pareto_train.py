import os
import argparse

cds = {
'cd':['3e-5', '6e-5', '1e-4', '2e-4', '3e-4'],
'cdt':['1e-5', '2e-5', '3e-5', '6e-5', '1e-4'],
}

def search_train():
    for cd in cds[opt.arg]:
        name = '%d_%s_'%(opt.it, opt.arg)+cd.replace('-','').replace('.','')
        os.system('python search_train.py --name %s --cd %s'%('f'+name, cd))

def main_train():
    for cd in cds[opt.arg]:
        name = '%d_%s_'%(opt.it, opt.arg)+cd.replace('-','').replace('.','')
        os.system('python main_train.py --name %s --mixm %s'%('x'+name, 'f'+name+'_last'))

parser = argparse.ArgumentParser()
parser.add_argument('--search', action='store_true')
parser.add_argument('--main', action='store_true')
parser.add_argument('--it', type=int)
parser.add_argument('--arg', type=str)
opt = parser.parse_args()

if opt.search:
    search_train()

if opt.main:
    main_train()
