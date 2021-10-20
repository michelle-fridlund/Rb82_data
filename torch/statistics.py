#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 13 2021 15:29
@author: michellef
"""
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

extent_stress_100 = [10, 30, 29, 0, 3, 0, 13, 27, 1, 16, 1]
extent_rest_100 = [1, 15, 0, 0, 2, 0, 2, 11, 0, 5, 0]
tpd_stress_100 = [8,22,21,1,2,0,10,20,1,12,1]
tpd_rest_100 = [2,10,0,0,1,0,2,8,0,6,0]

extent_stress_25 = [8,32,33,2,5,1,13,30,1,16,5]
extent_rest_25 = [2,15,0,0,4,0,4,10,0,5,1]
tpd_stress_25 = [6,23,24,2,4,1,10,22,2,13,4]
tpd_rest_25 = [1,11,0,0,2,0,3,8,0,5,1]

extent_stress_10 = [7,33,40,12,6,2,23,39,4,18,21]
extent_rest_10 = [2,17,3,3,4,0,8,16,1,8,2]
tpd_stress_10 = [6,23,29,9,5,2,16,29,4,13,16]
tpd_rest_10 = [2,12,3,2,2,0,6,12,1,5,2]

extent_stress_out = [2,31,31,0,2,0,11,27,0,11,0]
extent_rest_out = [1,14,0,0,2,0,1,9,0,5,0]
tpd_stress_out = [3,21,22,0,2,0,9,20,0,9,1]
tpd_rest_out = [1,9,0,0,2,0,1,7,0,4,0]


sv_stress_100 = [70,39,40,56,46,62,61,47,58,64,91]
sv_rest_100 = [54,30,38,53,39,35,23,46,33,51,85]
ef_stress_100 = [48,73,42,53,55,82,63,29,70,58,47]
ef_rest_100 = [48,65,61,72,54,79,44,40,71,71,52]

sv_stress_25 = [73,38,32,58,64,54,52,28,57,73,79]
sv_rest_25 = [52,31,34,30,54,33,51,49,35,35,65]
ef_stress_25 = [50,68,42,62,61,82,68,28,65,54,44]
ef_rest_25 = [48,68,52,69,52,77,56,35,67,53,45]

sv_stress_10 = [49,38,34,47,64,59,52,52,55,64,56]
sv_rest_10 = [41,28,22,34,48,32,24,45,42,29,72]
ef_stress_10 = [42,70,37,59,63,78,59,29,62,57,34]
ef_rest_10 = [41,57,51,68,62,79,41,38,94,46,38]

def corr(x, y, **kwargs):
    
    # Calculate the value
    coef = np.corrcoef(x, y)[0][1]
    # Make the label
    label = r'$\rho$ = ' + str(round(coef, 2))
    
    # Add the label to the plot
    ax = plt.gca()
    ax.annotate(label, xy = (1.5, 1.8), size = 10, xycoords = ax.transAxes)

df = pd.DataFrame(list(zip(extent_stress_100, extent_stress_25)),columns =['Extent_stress_FD', 'Extent_stress_QD'])
ax = sns.pairplot(df)
ax = ax.map_upper(corr)


plt.savefig('/homes/michellef/stats.png')