#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 11 2022 
@author: michellef
"""
import os
import xlrd
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from datetime import datetime


loc = ("/homes/michellef/rb82_doses.xlsx")
data_path = "/homes/michellef/my_projects/rb82_data/Dicoms_OCT8/100p_STAT"
file_name = "REST/Sinograms/REST-LM-00-sino-0.s.hdr"


# Extract indivudual column values from excel file
def read_x(loc):
    wb = xlrd.open_workbook(loc)
    sheet = wb.sheet_by_index(0)
    sheet.cell_value(0, 0)
    
    doses = []
    dates = []
    patients = []
    for i in range(sheet.nrows):
        doses.append(sheet.cell_value(i, 2))
        dates.append(sheet.cell_value(i, 1))
        patients.append(sheet.cell_value(i, 0))
    return doses, dates, patients


def plot(loc):
    doses, dates, patients = read_x(loc)
    dpd = pd.Series(doses)

    dpd.plot.hist(grid=False, bins=20, rwidth=0.9, color='#607c8e')
    plt.title('Rb-82 doses for 168 patients')
    plt.xlabel('Injected dose [MBq]')
    plt.ylabel('Patients')
    plt.savefig('/homes/michellef/clinical_eval/hist.png')
    plt.close("all")


def plot_dates(loc):
    doses, dates, patients = read_x(loc)

    df = pd.DataFrame(columns=['dates','doses'])
    for k,v in zip(doses, dates):
        df = df.append({'dates': v, 'doses':k},ignore_index=True)

    df['dates'] = pd.to_datetime(df['dates'], format='%Y%m%d')
    df.doses = df.doses.astype('float')
    ordered_df = df.sort_values(by='dates')

    pd.plotting.register_matplotlib_converters()
    plt.figure(figsize = (8,5))
    ax = sns.scatterplot(data = ordered_df, x = 'dates', y = 'doses', marker="x", color='crimson')
    ax.set(xlim = ('2016-05-01', '2020-08-01'))
    plt.title('Injected doses over time')
    plt.savefig('/homes/michellef/clinical_eval/scatter.png')
    plt.close("all")


def find_patients(loc):
    retained = []

    doses, dates, patients = read_x(loc)
    for k,v in zip(doses, patients):
        k = float(k)
        if k >= 1000.0 and k <= 1200.0:
            retained.append(v)
    
    print(f'{len(retained)} suitable patients found')
    return retained 

def get_counts(loc):
    patient_list = find_patients(loc)
    counts = []

    for p in patient_list:
        sino_path = os.path.join(data_path, p, file_name)
        with open(sino_path) as f:
            for line in f.readlines():
                line_ = line.strip()
                if line_.startswith('%total net trues:='):
                    c = line_.split(':=')[1]
                    counts.append(float(c))
    return counts

def plot_counts(loc):
    counts = get_counts(loc)
    
    dpd = pd.Series(counts)

    dpd.plot.hist(grid=False, bins=15, rwidth=0.9, color='#F55D3C')

    plt.title('Net trues for doses in the 1000-1200 MBq range')
    plt.xlabel('Total net trues')
    plt.savefig('/homes/michellef/clinical_eval/test.png')
    plt.close("all")

plot_counts(loc)


""" # Another method
import pyexcel
# Reading 
sheet = pyexcel.iget_records(file_name="excel.xlsx")
for row in sheet:
    print(row["Name"], row["Age"])
# Writing
data = [["Name", "Age"], ["Walker", 22], ["Leon", 24]]
sheet = pyexcel.Sheet(data, "Sheet1")
sheet.save_as("test.xlsx") """