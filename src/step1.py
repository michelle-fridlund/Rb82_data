#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  7 11:05:11 2020

@author: michellef
"""
import pickle
from datetime import datetime
from pathlib import Path
import glob
import os


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  6 12:11:17 2020

@author: michellef
"""
def update_pickle(patient):
    patient = pickle.load(open('patient.pkl','rb'))
    if len(LM) == 2 and len(PH) == 2:
                if not pt.name in p:
            p[pt.name] = {'CPR': CPRs[0],'ScanDate':DATEs[0]}
            p.patient = {
            "LISTMODE": LM,
            "PHYSIO": PH,
            "CALIBRATION": CL,
            "NORM": N,
            "OTHER": OTHER
                    }
    pickle.dump(p,open('patient.pkl','wb'))



def list_files(dir_path):
    filelist = []
    for (dirpath, dirnames, filenames) in os.walk(dir_path):
        filelist = glob.glob("{}/*.ptd".format(dirpath), recursive = True)
    print(filelist)


def id_files(dir_path):
    filelist = find_files(dir_path)

    for f in filelist:
        LM = []
        PH = []
        CL = []
        N = []
        OTHER = []
        if 'LISTMODE' in f.name or '.LM.' in f.name:
            LM.append(f)
        if 'PHYSIO' in f.name in f.name:
            PH.append(f)
        if 'CALIBRATION' in f.name:
            CL.append(f)
        if 'NORM' in f.name:
            N.append(f)
        else:
            OTHER.append(f)
    update+pickle()


def print_pt():
    p = pickle.load(open('patient.pkl','rb'))
    print(p.keys(), p.items())


if __name__ == "__main__":

    dir_path = '/homes/claes/projects/Lowdose_Rb82/RAWDATA'
    find_files(dir_path)
    print_pt()


