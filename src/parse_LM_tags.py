#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  6 12:11:17 2020
##############################################################################
Fetches unique tags from the LM and writes into pickle 
##############################################################################
@author: michellef
"""
import pickle
import os
import linecache
import argparse
import pandas as pd
from pathlib import Path
from ptd_parsers import ListmodeFileParser, PetctFileParser

patient_list = ['Al-Rammahi_Zainab_Fadil_20210324_111448','Mohammad_Harun_Younis_Youssef_20210622_090128',
'Rasmussen_Marcus_Brandt_20210407_122028', 'Stærmose_Lasse_Munkebo_Eriksen_20210412_110715',
'Pedersen_Laura_Søderberg_20210701_141631', 'Brun_Tobias_20210625_100649', 'Bagger_Gustav_Johan_Kolding_20210203_122643',
'Jensen_Alper_20210316_115155', 'Czadowska_Julia_Marcelina_20210622_085956', 'Jensen_Rebekka_Tenna_20210604_162115',
'Ravnsbjerg_Emilie_Hegelund_20210407_121909']


# Find patients with LM files
def find_LM(dir_path):
    pt = Path(dir_path)
    if not pt.is_dir():
        return None
    for f in pt.iterdir():
        if 'LISTMODE' in f.name or '.LM.' in f.name:
            return f
    return None


# Fetch LM info
def get_dump(new_path):
    LM = find_LM(new_path)
    if LM:
        os.system(f'strings "{LM}" | tail -200 > "{new_path}"/dump.txt')
        

# Find patients and get info from LISTMODE files
def find_patients(dir_path):
    patients = {}
    #for (dirpath, dirnames, filenames) in os.walk(dir_path):
    #    dirname = str(Path(dirpath).relative_to(dir_path))
    #    new_path = Path(os.path.join(dir_path, dirname))
        #get_dump(new_path)
    for p in patient_list:
        new_path = Path(os.path.join(dir_path, p))
        if (new_path/'dump.txt').exists():
            with open(new_path/'dump.txt') as f:
                for i, line in enumerate(f.readlines()):
                    # StudyInstance is before PETCT tag
                    if line.startswith('PETCT'):
                        # Get line before and remove \n
                        prev_line = linecache.getline(
                            f'{str(new_path)}/dump.txt', i).strip()
                        # Return patient name and StudyInstanceUID
                        patients[p] = prev_line
                        #patients[dirname] = prev_line
    return patients


# Find SeriesInstance UID and create a final patient dictionary
def get_uid(dir_path):
    pickle_info = {}
    patients = find_patients(dir_path)

    for k, v in patients.items():
        # Patient directories with appropriate files
        new_path = os.path.join(dir_path, k)

        parser1 = ListmodeFileParser(new_path)
        # parse listmode file into a list of lines
        parser1.read_tail(stopword='DICM')
        # parse lines containing := into dictionnary
        parser1.get_primary_info(include='=', exclude='!')
        # parse the rest of the info (where values aren't labeled (no :=))
        parser1.get_secondary_info()
        # turn values to int if possible
        parser1.clean_info()
        
        """ PARSE PETCT FILE FOR SeriesInstanceUID"""
        parser2 = PetctFileParser(new_path)
        # skip patients without CT file
        if parser2.file_in:
            parser2.read_tail(stopword='CTSeriesDicomUIDforAC')
            parser2.get_primary_info()
            parser2.clean_info()
        
        pickle_info[k] = {**parser1.info, **parser2.info}

    return pickle_info

"""         # We can only get SeriesInstanceUID from PETCT_SPL files
        parser = PetctFileParser(new_path)  # Raphael's parser

        if parser.file_in:
            parser.read_tail(stopword='CTSeriesDicomUIDforAC')
            parser.get_primary_info()
            parser.clean_info()

        if parser.info['CTSeriesDicomUIDforAC'] != '':
            pickle_info[k] = {
                'StudyInstance': v, 'SeriesInstance': parser.info['CTSeriesDicomUIDforAC']} """


# Write dump files
def write_pickle(dir_path):
    patients = get_uid(dir_path)
    df = pd.DataFrame(patients).transpose()

    print(f'{len(patients.keys())} PATIENTS FOUND!')

    # Drop mising StudyInstance
    index = df.index
    condition = df['StudyInstance'] == 'Biograph64_mCT-1104'
    condition2 = df['CTSeriesDicomUIDforAC'] == ''
    missing_input = index[condition]
    df.drop(missing_input, inplace=True)

    df.dropna(subset=['CTSeriesDicomUIDforAC'], inplace=True)
    # drop duplicates based on study date and time
    subset_for_comparison = ['StudyInstanceUID', 
                            'CTSeriesDicomUIDforAC', 
                            "study time (hh:mm:ss GMT+00:00)", 
                            "study date (yyyy:mm:dd)", 
                            "name of data file"]
    df.drop_duplicates(subset=subset_for_comparison, keep='last', inplace=True)

    df_for_ct_match = df[['StudyInstanceUID', 'CTSeriesDicomUIDforAC']]

    os.chdir(dir_path)
    with open('paediatrics_studyuid2.pickle', 'wb') as p:
        pickle.dump(df_for_ct_match, p)

    print(pickle.load(open('paediatrics_studyuid2.pickle', 'rb')))

    # Nice print statement
    #print(df)


if __name__ == "__main__":
    # Initiate the parser
    parser = argparse.ArgumentParser()
    required_args = parser.add_argument_group('required arguments')

    # Required args: LM data path
    required_args.add_argument(
        "--data", "-d", dest='data',  help="patient directory", required=True)

    # Read arguments from the command line
    args = parser.parse_args()

    dir_path = Path(args.data)

    write_pickle(dir_path)
