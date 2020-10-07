#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  6 12:11:17 2020

@author: michellef
"""
import pickle, os
from datetime import datetime
from pathlib import Path
def query_ptd(pt):
    CPRs = []
    DATEs = []
    
    with open(pt/'dump.txt') as f:
        for line in f.readlines():
            line_ = line.strip()
            if len(line_) == 10:
                try:
                    int(line_)
                    CPRs.append(line_)
                except:
                    continue
            if line_.startswith('%tracer injection date (yyyy:mm:dd):='):
                d = line_.split(':=')[1]
                scandate = datetime.strptime(d,'%Y:%m:%d').strftime('%Y%m%d')
                DATEs.append(scandate)
    update_pickle(pt,CPRs,DATEs)

def update_pickle(pt,CPRs,DATEs):
    p = pickle.load(open('scaninfo.pkl','rb'))
    if len(CPRs) == 1:
        #k = f'{CPRs[0]}_{DATEs[0]}'
        if not pt.name in p:
            p[pt.name] = {'CPR': CPRs[0],'ScanDate':DATEs[0]}
    pickle.dump(p,open('scaninfo.pkl','wb'))

def find_LM(pt):
    if not pt.is_dir():
        return None
    for f in pt.iterdir():
        if 'LISTMODE' in f.name or '.LM.' in f.name:
            return f
    return None

def build_dump(pt):
    LM = find_LM(pt)
    if LM:
        print(f'strings "{LM.absolute()}" | tail -200 > "{pt}"/dump.txt')
        os.system(f'strings "{LM.absolute()}" | tail -200 > "{pt}"/dump.txt')
        
def print_pkl():
    p = pickle.load(open('scaninfo.pkl','rb'))
    print(p.keys(), p.items())

for pt in Path('.').iterdir():
    if not (pt/'dump.txt').exists():
        build_dump(pt)
    if (pt/'dump.txt').exists():
        query_ptd(pt)

print_pkl()
