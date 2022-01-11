#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 11 2022 
@author: michellef
"""
import xlrd
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from datetime import datetime


loc = ("/homes/michellef/rb82_doses.xlsx")

def read_x(loc):
    wb = xlrd.open_workbook(loc)
    sheet = wb.sheet_by_index(0)
    sheet.cell_value(0, 0)
    
    doses = []
    dates = []
    for i in range(sheet.nrows):
        doses.append(sheet.cell_value(i, 2))
        dates.append(sheet.cell_value(i, 1))

    return doses, dates

def plot(loc):
    doses, dates = read_x(loc)
    dpd = pd.Series(doses)

    dpd.plot.hist(grid=False, bins=20, rwidth=0.9, color='#607c8e')
    plt.title('Rb-82 doses for 168 patients')
    plt.xlabel('Injected dose [MBq]')
    plt.ylabel('Patients')
    plt.savefig('/homes/michellef/clinical_eval/test.png')
    plt.close("all")

def plot_dates(loc):
    doses, dates = read_x(loc)

    df = pd.DataFrame(columns=['dates','doses'])
    for k,v in zip(doses, dates):
        df = df.append({'dates': v, 'doses':k},ignore_index=True)

    df['dates'] = pd.to_datetime(df['dates'], format='%Y%m%d')
    df.doses = df.doses.astype('float')
    ordered_df = df.sort_values(by='dates')
    
    #df.plot.scatter(x='dates', y='doses')
    pd.plotting.register_matplotlib_converters()
    plt.figure(figsize = (8,5))
    ax = sns.scatterplot(data = ordered_df, x = 'dates', y = 'doses', marker="x", color='crimson')
    ax.set(xlim = ('2016-05-01', '2020-08-01'))
    plt.title('Injected doses over time')
    plt.savefig('/homes/michellef/clinical_eval/scatter.png')
    plt.close("all")

plot_dates(loc)