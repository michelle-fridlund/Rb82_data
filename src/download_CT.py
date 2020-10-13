#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  3 14:52:57 2020

@author: claes
"""

import re
import pickle
from pydicom.dataset import Dataset
from pynetdicom import AE, evt, build_role, debug_logger
from pynetdicom.sop_class import (
    PatientRootQueryRetrieveInformationModelGet,
    PatientRootQueryRetrieveInformationModelFind,
    CTImageStorage,
    MRImageStorage,
    PositronEmissionTomographyImageStorage
)
from pathlib import Path
#debug_logger()

# Implement the handler for evt.EVT_C_STORE
def handle_store(event,todir='.'):
    """Handle a C-STORE request event."""
    ds = event.dataset
    ds.file_meta = event.file_meta

    saved_file_name = f'{todir}/{ds.SOPInstanceUID}'
    ds.save_as(saved_file_name, write_like_original=False)

    # Return a 'Success' status
    return 0x0000


def get_dataset(ID):
    # Create our Identifier (query) dataset
    # We need to supply a Unique Key Attribute for each level above the
    #   Query/Retrieve level
    ds = Dataset()
    ds.QueryRetrieveLevel = 'SERIES'
    ds.PatientID = ID.split('_')[0]
    ds.StudyInstanceUID = dali_datasets[ID]['StudyInstanceUID']
    ds.SeriesInstanceUID = dali_datasets[ID]['SeriesInstanceUID']
    return ds

"""
  - SeriesDescription
  - SeriesInstanceUID
  - Model
  - InstitutionalDepartmentName
  - InstitutionName
  - StationName
  - StudyInstanceUID
"""
dali_hits = pickle.load(open('MYDATA_OCT8/scaninfo.pkl','rb'))

def download_CT(ds,ID):
    
    """
    Setup the connection
    """
    PatientID = ID.split('_')[0]
    StudyDate = ID.split('_')[1]

    # Target directory
    save_to_dir = Path(f'retrieved_data/{name}{ID}/CT')
    for int_suffix in range(20): 
        if save_to_dir.exists():
            save_to_dir = Path(f'retrieved_data/{name}{ID}/CT_{int_suffix}')
        else:
            pass
    save_to_dir.mkdir(parents=True)

    #todir = f'retrieved_data/{PatientID}_{StudyDate}/CT'
    handlers = [(evt.EVT_C_STORE, lambda x: handle_store(event=x,todir=save_to_dir))]
    """
    Setup type of data to download:
        PositronEmissionTomographyImageStorage for PET
        CTImageStorage for CT?
    """
    #aeFIND.add_requested_context(CTImageStorage)
    role = build_role(CTImageStorage, scp_role=True)
    
    assoc = aeFIND.associate(addr='dali', port=11112, ext_neg=[role], ae_title="DALI", evt_handlers=handlers)
    if assoc.is_established:
        # Use the C-GET service to send the identifier
        responses = assoc.send_c_get(ds, PatientRootQueryRetrieveInformationModelGet)
        for (status, identifier) in responses:
            if status:
                print('C-GET query status: 0x{0:04x}'.format(status.Status))
            else:
                print('Connection timed out, was aborted or received invalid response')
    
        # Release the association
        assoc.release()
    else:
        print('Association rejected, aborted or never connected')

aeFIND = AE('MN240849')
aeFIND.add_requested_context(PatientRootQueryRetrieveInformationModelFind)

#aeGET = AE('MN240849')
aeFIND.add_requested_context(PatientRootQueryRetrieveInformationModelGet)
aeFIND.add_requested_context(CTImageStorage)

# def add_seq(identifier,ID):
#     global dali_datasets
#     if not ID in dali_datasets:
#         dali_datasets[ID] = {}
#         dali_datasets[ID]['StudyInstanceUID'] = identifier.StudyInstanceUID        
#         dali_datasets[ID]['SeriesDescription'] = []
#         dali_datasets[ID]['SeriesInstanceUID'] = []
#     dali_datasets[ID]['SeriesDescription'].append(identifier.SeriesDescription)
#     dali_datasets[ID]['SeriesInstanceUID'].append(identifier.SeriesInstanceUID)
    

def dali(ID):
    
    ds = Dataset()
    ds.QueryRetrieveLevel = 'SERIES'
    ds.SeriesDescription = '*'
    ds.StudyInstanceUID = '*'
    ds.Modality = 'CT'
    ds.PatientID = ID.split('_')[0]
    ds.StudyDate = ID.split('_')[1]
    ds.SeriesInstanceUID = '*'

    assoc = aeFIND.associate(addr='dali', port=11112, ae_title="DALI")
    if not assoc.is_established:
        print("NOT ESTABLISHED")
        exit(-1)
    responses = assoc.send_c_find(ds, query_model=PatientRootQueryRetrieveInformationModelFind)
    
    for (status, identifier) in responses:

        if(status.Status>0):
            print(identifier)
            # add_seq(identifier,ID)
            download_CT(identifier,ID)
                        
    assoc.release()

"""
Loop patients and download
"""
dali_datasets = {}
# Get a patient
for k,v in dali_hits.items():
    k_name = re.findall('^[^0-9]*', k) #PATIENT NAME ADDED FOR SIMPLICITY, MF
    name = k_name[0]
    ID = '{}_{}'.format(v['CPR'],v['ScanDate'])
    if not Path(f'retrieved_data/{name}_{ID}/CT').exists():

#        print(f"Downloading data for {name}_{ID}")        
        dali(ID)
