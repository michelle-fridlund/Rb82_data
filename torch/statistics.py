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
from scipy import stats

patients = ['6db1c5da-8e89-4287-8a8b-57c64ed0109b', '6fa399c0-ad77-4e0c-bea8-37ce8ab36dbf', '28c4f1c2-2cbb-489e-ab57-09b13a60231c', '38bcb821-3c99-4eb1-95e5-2a5d001193d3',
'3616f6a0-b08a-4253-b072-431f699f5886',
'4019e4ab-b8bd-40b3-bb1d-f176feb13384',
'b48007ee-dbe4-4fe9-a7e2-7c3020f468b7',
'0ef7e890-6586-4876-a630-a3af8e7fd736',
'7b4979e9-767d-422c-9cec-f80f787ded49',
'500cfb2b-e287-4e4c-8143-273524f7564b',
'2b6a889f-be7b-473c-a941-758fe8187fa9'
]

# Static
extent_stress_100 = [10, 30, 29, 0, 3, 0, 13, 27, 1, 16, 1]
extent_rest_100 = [1, 15, 0, 0, 2, 0, 2, 11, 0, 5, 0]
tpd_stress_100 = [8,22,21,1,2,0,10,20,1,12,1]
tpd_rest_100 = [2,10,0,0,1,0,2,8,0,6,0]

extent_stress_25 = [8,32,33,2,5,1,13,30,1,16,5]
extent_rest_25 = [2,15,0,0,4,0,4,10,0,5,1]
tpd_stress_25 = [6,23,24,2,4,1,10,22,2,13,4]
tpd_rest_25 = [1,11,0,0,2,0,3,8,0,5,1]

extent_stress_12 = [10, 34, 34, 8, 7, 2, 17, 33, 3, 24, 12]
extent_rest_12 = [3, 17, 2, 2, 3, 0, 5, 13, 1, 10, 2]
tpd_stress_12 = [8, 22, 25, 6, 5, 2, 13, 25, 3, 16, 10]
tpd_rest_12 = [3, 12, 2, 1, 2, 0, 4, 9, 1, 7, 2]

extent_stress_10 = [7,33,40,12,6,2,23,39,4,18,21]
extent_rest_10 = [2,17,3,3,4,0,8,16,1,8,2]
tpd_stress_10 = [6,23,29,9,5,2,16,29,4,13,16]
tpd_rest_10 = [2,12,3,2,2,0,6,12,1,5,2]

# ResNet_residual
extent_stress_out = [2,31,31,0,2,0,11,27,0,11,0]
extent_rest_out = [1,14,0,0,2,0,1,9,0,5,0]
tpd_stress_out = [3,21,22,0,2,0,9,20,0,9,1]
tpd_rest_out = [1,9,0,0,2,0,1,7,0,4,0]


# Gated
sv_stress_100 = [70,39,40,56,46,62,61,47,58,64,91]
sv_rest_100 = [54,30,38,53,39,35,23,46,33,51,85]
ef_stress_100 = [48,73,42,53,55,82,63,29,70,58,47]
ef_rest_100 = [48,65,61,72,54,79,44,40,71,71,52]

sv_stress_25 = [73,38,32,58,64,54,52,28,57,73,79]
sv_rest_25 = [52,31,34,30,54,33,51,49,35,35,65]
ef_stress_25 = [50,68,42,62,61,82,68,28,65,54,44]
ef_rest_25 = [48,68,52,69,52,77,56,35,67,53,45]

sv_stress_12 = [65, 40, 20, 56, 44, 54, 48, 32, 49,59, 66]
sv_rest_12 = [45, 28, 32, 35, 47, 31, 34, 37, 33, 36, 80]
ef_stress_12 = [47, 73, 32, 61, 55, 82, 56, 26, 68, 56, 24]
ef_rest_12 = [43,61,55,67,55,74,50,34,69,49,41]

sv_stress_10 = [49,38,34,47,64,59,52,52,55,64,56]
sv_rest_10 = [41,28,22,34,48,32,24,45,42,29,72]
ef_stress_10 = [42,70,37,59,63,78,59,29,62,57,34]
ef_rest_10 = [41,57,51,68,62,79,41,38,94,46,38]


# Dynamic
mbf_stress_100 = [2.98,3.01,1.66,3.66,3.32,4.34,2.52,2.13,3.98,2.78,2.03]
mbf_rest_100 = [1.42,1.59,1.25,2,1.36,1.5,1.31,1.56,1.4,1.06,1.39]
mfr_100 = [2.11,1.89,1.36,1.82,2.52,2.91,1.94,1.33,2.88,2.66,1.47]

mbf_stress_25 = [2.79,3.04,1.87,3.18,3.09,3.96,2.47,2.17,3.71,2.63,2.16]
mbf_rest_25 = [1.29,1.28,1.17,1.44,1.24,1.32,1.27,1.26,1.25,1.02,1.29]
mfr_25 = [2.09,2.36,1.62,2.2,2.51,3,1.94,1.67,2.99,2.66,1.69]

mbf_stress_12 = [2.9,2.57,1.58,3.01,3.65,3.9,2.57,1.8,3.87,2.78,2.44]
mbf_rest_12 = [1.25,1.27,1.14,1.34,1.22,1.26,1.11,1.19,1.22,1.14,1.25]
mfr_12 = [2.32,2.03,1.4,2.23,3.01,3.09,2.36,1.47,3.19,2.56,1.94]

mbf_stress_10 = [3.05,2.75,1.5,2.84,3.46,3.78,2.59,1.63,3.57,2.44,2.36]
mbf_rest_10 = [1.26,1.31,1.02,1.31,1.24,1.29,1.16,1.23,1.22,1.01,1.28]
mfr_10 = [2.42,2.1,1.47,2.15,2.78,2.92,2.31,1.31,2.94,2.46,0.33]

edv_stress_100 = [145,53,94,104,82,76,98,165,83,111,195]
edv_stress_25 = [144,56,76,93,106,66,76,100,87,135,180]
edv_stress_12 = [138,55,62,91,80,66,85,124,73,106,272]
edv_stress_10 = [118,54,93,79,102,75,87,180,90,113,166]

edv_rest_100 = [112,46,63,73,72,44,53,115,46,72,163]
edv_rest_25 = [109,46,66,72,106,43,91,139,52,66,145]
edv_rest_12 = [105,46,58,53,85,42,68,107,48,74,193]
edv_rest_10 = [101,50,43,64,77,41,59,120,45,63,191]

esv_stress_100 = [75,14,54,49,37,14,37,118,25,47,104]
esv_stress_25 = [72,18,44,35,41,12,25,72,30,61,101]
esv_stress_12 = [74,15,43,35,36,12,37,92,24,47,206]
esv_stress_10 = [69,16,59,33,38,17,35,128,34,49,110]

esv_rest_100 = [58,16,24,20,33,9,30,69,13,21,77]
esv_rest_25 = [57,15,31,22,51,10,40,90,17,31,80]
esv_rest_12 = [60,18,26,18,38,11,34,70,15,37,113]
esv_rest_10 = [60,22,21,21,29,8,35,75,13,34,119]


df2 = pd.DataFrame({'tpd_stress_100': [8,22,21,1,2,0,10,20,1,12,1],
                    'tpd_rest_100': [2,10,0,0,1,0,2,8,0,6,0],
                    'tpd_stress_25': [6,23,24,2,4,1,10,22,2,13,4],
                    'tpd_rest_25': [1,11,0,0,2,0,3,8,0,5,1],
                    'tpd_stress_12': [8, 22, 25, 6, 5, 2, 13, 25, 3, 16, 10],
                    'tpd_rest_12': [3, 12, 2, 1, 2, 0, 4, 9, 1, 7, 2],
                    'tpd_stress_10': [6,23,29,9,5,2,16,29,4,13,16],
                    'tpd_rest_10': [2,12,3,2,2,0,6,12,1,5,2],
                    'edv_stress_100': [145,53,94,104,82,76,98,165,83,111,195],
                    'edv_stress_25': [144,56,76,93,106,66,76,100,87,135,180],
                    'edv_stress_12': [138,55,62,91,80,66,85,124,73,106,272],
                    'edv_stress_10': [118,54,93,79,102,75,87,180,90,113,166],
                    'edv_rest_100': [112,46,63,73,72,44,53,115,46,72,163],
                    'edv_rest_25': [109,46,66,72,106,43,91,139,52,66,145],
                    'edv_rest_12': [105,46,58,53,85,42,68,107,48,74,193],
                    'edv_rest_10': [101,50,43,64,77,41,59,120,45,63,191],
                    'esv_stress_100': [75,14,54,49,37,14,37,118,25,47,104],
                    'esv_stress_25': [72,18,44,35,41,12,25,72,30,61,101],
                    'esv_stress_12': [74,15,43,35,36,12,37,92,24,47,206],
                    'esv_stress_10': [69,16,59,33,38,17,35,128,34,49,110],
                    'esv_rest_100': [58,16,24,20,33,9,30,69,13,21,77],
                    'esv_rest_25': [57,15,31,22,51,10,40,90,17,31,80],
                    'esv_rest_12': [60,18,26,18,38,11,34,70,15,37,113],
                    'esv_rest_10': [60,22,21,21,29,8,35,75,13,34,119]},
                    index=['6db1c5da-8e89-4287-8a8b-57c64ed0109b', 
                           '6fa399c0-ad77-4e0c-bea8-37ce8ab36dbf', 
                           '28c4f1c2-2cbb-489e-ab57-09b13a60231c', 
                           '38bcb821-3c99-4eb1-95e5-2a5d001193d3',
                           '3616f6a0-b08a-4253-b072-431f699f5886',
                           '4019e4ab-b8bd-40b3-bb1d-f176feb13384',
                           'b48007ee-dbe4-4fe9-a7e2-7c3020f468b7',
                           '0ef7e890-6586-4876-a630-a3af8e7fd736',
                           '7b4979e9-767d-422c-9cec-f80f787ded49',
                           '500cfb2b-e287-4e4c-8143-273524f7564b',
                           '2b6a889f-be7b-473c-a941-758fe8187fa9'])

df = pd.DataFrame(columns=['clinical','low','dose'])
for k,v in zip(esv_rest_100,esv_rest_25):
    df = df.append({'clinical': k, 'low':v, 'dose':'25%'},ignore_index=True)
for k,v in zip(esv_rest_100,esv_rest_12): # HERE!!!
    df = df.append({'clinical': k, 'low':v, 'dose':'12%'},ignore_index=True) # HERE!!!
df.clinical = df.clinical.astype('float')
df.low = df.low.astype('float')

#line_kws={'label':"y={0:.2f}x+{1:.2f}".format(slope,intercept)}    # HERE!!!
slope, intercept, r_value, p_value, std_err = stats.linregress(df2['esv_rest_25'],df2['esv_rest_100'])  
slope2, intercept2, r_value2, p_value2, std_err2 = stats.linregress(df2['esv_rest_12'],df2['esv_rest_100']) # HERE!!!
print(p_value, p_value2)

fgrid = sns.lmplot(x="clinical", y="low", data=df, hue = 'dose', palette="Set1")
ax = fgrid.axes[0,0]   # HERE!!! p={0:.4f}".format(p_value)
#ax.set(xlabel="Clinical LVEF stress [mL/(min·g)]", ylabel="Low-Dose MBF rest [mL/(min·g)]") 
ax.set(xlabel="Clinical ESV rest [mL]", ylabel="Low-Dose ESV rest [mL]") 
plt.text(20, 100, "y={0:.2f}x{1:.2f}".format(slope,intercept), horizontalalignment='left', size='medium', color='crimson', weight='semibold')
plt.text(20, 92, "R={0:.2f}".format(r_value), horizontalalignment='left', size='medium', color='crimson', weight='semibold')
plt.text(20, 84, "p=1.88e-06", horizontalalignment='left', size='medium', color='crimson', weight='semibold')
plt.text(55, 25, "y={0:.2f}x+{1:.2f}".format(slope2,intercept2), horizontalalignment='left', size='medium', color='steelblue', weight='semibold')
plt.text(55, 17, "R={0:.2f}".format(r_value2), horizontalalignment='left', size='medium', color='steelblue', weight='semibold')
plt.text(55, 9, "p=6.05e-06", horizontalalignment='left', size='medium', color='steelblue', weight='semibold')
plt.savefig('/homes/michellef/clinical_eval/TEST.png')