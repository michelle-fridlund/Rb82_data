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
'500cfb2b-e287-4e4c-8143-273524f7564b']

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

extent_stress_5 = [21,35,38,20,11,10,36,38,15,27,32]
extent_rest_5 = [8,20,11,5,5,0,18,18,4,12,9]
tpd_stress_5 = [15,24,28,15,9,8,25,28,10,19,22]
tpd_rest_5 = [6,15,7,4,4,1,12,13,3,8,7]

# ResNet_residual
extent_stress_out = [2,31,31,0,2,0,11,27,0,11,0]
extent_rest_out = [1,14,0,0,2,0,1,9,0,5,0]
tpd_stress_out = [3,21,22,0,2,0,9,20,0,9,1]
tpd_rest_out = [1,9,0,0,2,0,1,7,0,4,0]

# 2mm/2mm
extent_stress_2mm = [5,37,31,1,2,0,14,29,2,15]
extent_rest_2mm = [1,15,0,0,2,0,3,10,0,4]
tpd_stress_2mm = [5,26,22,1,2,1,11,22,1,12]
tpd_rest_2mm = [1,11,0,0,1,0,3,8,0,5]

# Gated
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

sv_stress_100 = [70,39,40,56,46,62,61,47,58,64,91]
sv_rest_100 = [54,30,38,53,39,35,23,46,33,51,85]
ef_stress_100 = [48,73,42,53,55,82,63,29,70,58]
ef_rest_100 = [48,65,61,72,54,79,44,40,71,71]

sv_stress_25 = [73,38,32,58,64,54,52,28,57,73,79]
sv_rest_25 = [52,31,34,30,54,33,51,49,35,35,65]
ef_stress_25 = [50,68,42,62,61,82,68,28,65,54]
ef_rest_25 = [48,68,52,69,52,77,56,35,67,53]

sv_stress_12 = [65, 40, 20, 56, 44, 54, 48, 32, 49,59, 66]
sv_rest_12 = [45, 28, 32, 35, 47, 31, 34, 37, 33, 36, 80]
ef_stress_12 = [47, 73, 32, 61, 55, 82, 56, 26, 68, 56]
ef_rest_12 = [43,61,55,67,55,74,50,34,69,49]

sv_stress_10 = [49,38,34,47,64,59,52,52,55,64,56]
sv_rest_10 = [41,28,22,34,48,32,24,45,42,29,72]
ef_stress_10 = [42,70,37,59,63,78,59,29,62,57]
ef_rest_10 = [41,57,51,68,62,79,41,38,94,46]

edv_stress_5 = [110,47,75,74,100,60,93,184,73,99,210]
edv_rest_5 = [123,44,60,61,72,39,87,115,40,83,146]
esv_stress_5 = [68,14,53,24,41,12,45,144,27,54,149]
esv_rest_5 = [68,18,28,22,29,9,7,74,20,13,89]
sv_stress_5 = [42,33,21,50,59,48,47,39,47,44,60]
sv_rest_5 = [55,27,31,39,44,30,80,41,20,70,57]
ef_stress_5 = [38,70,29,67,59,80,51,22,64,45]
ef_rest_5 = [45,60,53,64,60,77,92,36,50,84]


# Dynamic
mbf_stress_100 = [2.98,3.01,1.66,3.66,3.32,4.34,2.52,2.13,3.98,2.78]
mbf_rest_100 = [1.42,1.59,1.25,2,1.36,1.5,1.31,1.56,1.4,1.06]
mfr_100 = [2.11,1.89,1.36,1.82,2.52,2.91,1.94,1.33,2.88,2.66]

mbf_stress_25 = [2.79,3.04,1.87,3.18,3.09,3.96,2.47,2.17,3.71,2.63]
mbf_rest_25 = [1.29,1.28,1.17,1.44,1.24,1.32,1.27,1.26,1.25,1.02]
mfr_25 = [2.09,2.36,1.62,2.2,2.51,3,1.94,1.67,2.99,2.66]

mbf_stress_12 = [2.9,2.57,1.58,3.01,3.65,3.9,2.57,1.8,3.87,2.78]
mbf_rest_12 = [1.25,1.27,1.14,1.34,1.22,1.26,1.11,1.19,1.22,1.14]
mfr_12 = [2.32,2.03,1.4,2.23,3.01,3.09,2.36,1.47,3.19,2.56]

mbf_stress_10 = [3.05,2.75,1.5,2.84,3.46,3.78,2.59,1.63,3.57,2.44]
mbf_rest_10 = [1.26,1.31,1.02,1.31,1.24,1.29,1.16,1.23,1.22,1.01]
mfr_10 = [2.42,2.1,1.47,2.15,2.78,2.92,2.31,1.31,2.94,2.46]

mbf_stress_5 = [3.34,3.08,1.97,3.13,2.65,3.94,2.48,1.55,3.56,3.1]
mbf_rest_5 = [1.27,1.23,1.1,1.3,1.07,1.16,1,1.13,1.11,1.08]
mfr_5 = [2.63,2.5,1.8,2.41,2.54,3.55,2.57,1.34,3.27,2.97]



df2 = pd.DataFrame({'tpd_stress_100': [8,22,21,1,2,0,10,20,1,12],
                    'tpd_rest_100': [2,10,0,0,1,0,2,8,0,6],
                    'tpd_stress_25': [6,23,24,2,4,1,10,22,2,13],
                    'tpd_rest_25': [1,11,0,0,2,0,3,8,0,5],
                    'tpd_stress_12': [8, 22, 25, 6, 5, 2, 13, 25, 3, 16],
                    'tpd_rest_12': [3, 12, 2, 1, 2, 0, 4, 9, 1, 7],
                    'tpd_stress_10': [6,23,29,9,5,2,16,29,4,13],
                    'tpd_rest_10': [2,12,3,2,2,0,6,12,1,5],
                    'tpd_stress_2mm': [5,26,22,1,2,1,11,22,1,12],
                    'tpd_rest_2mm': [1,11,0,0,1,0,3,8,0,5],
                    'edv_stress_100': [145,53,94,104,82,76,98,165,83,111],
                    'edv_stress_25': [144,56,76,93,106,66,76,100,87,135],
                    'edv_stress_12': [138,55,62,91,80,66,85,124,73,106],
                    'edv_stress_10': [118,54,93,79,102,75,87,180,90,113],
                    'edv_rest_100': [112,46,63,73,72,44,53,115,46,72],
                    'edv_rest_25': [109,46,66,72,106,43,91,139,52,66],
                    'edv_rest_12': [105,46,58,53,85,42,68,107,48,74],
                    'edv_rest_10': [101,50,43,64,77,41,59,120,45,63],
                    'esv_stress_100': [75,14,54,49,37,14,37,118,25,47],
                    'esv_stress_25': [72,18,44,35,41,12,25,72,30,61],
                    'esv_stress_12': [74,15,43,35,36,12,37,92,24,47],
                    'esv_stress_10': [69,16,59,33,38,17,35,128,34,49],
                    'esv_rest_100': [58,16,24,20,33,9,30,69,13,21],
                    'esv_rest_25': [57,15,31,22,51,10,40,90,17,31],
                    'esv_rest_12': [60,18,26,18,38,11,34,70,15,37],
                    'esv_rest_10': [60,22,21,21,29,8,35,75,13,34],
                    'ef_stress_100': [48,73,42,53,55,82,63,29,70,58],
                    'ef_rest_100': [48,65,61,72,54,79,44,40,71,71],
                    'ef_stress_25': [50,68,42,62,61,82,68,28,65,54],
                    'ef_rest_25': [48,68,52,69,52,77,56,35,67,53],
                    'ef_stress_10': [42,70,37,59,63,78,59,29,62,57],
                    'ef_rest_10': [41,57,51,68,62,79,41,38,94,46],
                    'ef_stress_5': [38,70,29,67,59,80,51,22,64,45],
                    'ef_rest_5': [45,60,53,64,60,77,92,36,50,84],
                    'mbf_stress_100': [2.98,3.01,1.66,3.66,3.32,4.34,2.52,2.13,3.98,2.78],
                    'mbf_rest_100': [1.42,1.59,1.25,2,1.36,1.5,1.31,1.56,1.4,1.06],
                    'mfr_100': [2.11,1.89,1.36,1.82,2.52,2.91,1.94,1.33,2.88,2.66],
                    'mbf_stress_25': [2.79,3.04,1.87,3.18,3.09,3.96,2.47,2.17,3.71,2.63],
                    'mbf_rest_25': [1.29,1.28,1.17,1.44,1.24,1.32,1.27,1.26,1.25,1.02],
                    'mfr_25': [2.09,2.36,1.62,2.2,2.51,3,1.94,1.67,2.99,2.66],
                    'mbf_stress_10': [3.05,2.75,1.5,2.84,3.46,3.78,2.59,1.63,3.57,2.44],
                    'mbf_rest_10': [1.26,1.31,1.02,1.31,1.24,1.29,1.16,1.23,1.22,1.01],
                    'mfr_10': [2.42,2.1,1.47,2.15,2.78,2.92,2.31,1.31,2.94,2.46],
                    'mbf_stress_5': [3.34,3.08,1.97,3.13,2.65,3.94,2.48,1.55,3.56,3.1],
                    'mbf_rest_5': [1.27,1.23,1.1,1.3,1.07,1.16,1,1.13,1.11,1.08],
                    'mfr_5': [2.63,2.5,1.8,2.41,2.54,3.55,2.57,1.34,3.27,2.97]},
                    index=['6db1c5da-8e89-4287-8a8b-57c64ed0109b', 
                           '6fa399c0-ad77-4e0c-bea8-37ce8ab36dbf', 
                           '28c4f1c2-2cbb-489e-ab57-09b13a60231c', 
                           '38bcb821-3c99-4eb1-95e5-2a5d001193d3',
                           '3616f6a0-b08a-4253-b072-431f699f5886',
                           '4019e4ab-b8bd-40b3-bb1d-f176feb13384',
                           'b48007ee-dbe4-4fe9-a7e2-7c3020f468b7',
                           '0ef7e890-6586-4876-a630-a3af8e7fd736',
                           '7b4979e9-767d-422c-9cec-f80f787ded49',
                           '500cfb2b-e287-4e4c-8143-273524f7564b'])

df = pd.DataFrame(columns=['clinical','low','dose'])
for k,v in zip(tpd_stress_100,tpd_stress_25):
    df = df.append({'clinical': k, 'low':v, 'dose':'25%'},ignore_index=True)
for k,v in zip(tpd_stress_100,tpd_stress_2mm): # HERE!!!
    df = df.append({'clinical': k, 'low':v, 'dose':'25% (Denoised)'},ignore_index=True) # HERE!!!
df.clinical = df.clinical.astype('float')
df.low = df.low.astype('float')

#line_kws={'label':"y={0:.2f}x+{1:.2f}".format(slope,intercept)}    # HERE!!!
""" slope, intercept, r_value, p_value, std_err = stats.linregress(df2['tpd_stress_100'],df2['tpd_stress_25'])  
slope2, intercept2, r_value2, p_value2, std_err2 = stats.linregress(df2['tpd_stress_100'],df2['tpd_stress_2mm']) # HERE!!!
print(p_value, p_value2)

fgrid = sns.lmplot(x="clinical", y="low", data=df, hue = 'dose', palette="Set1")
ax = fgrid.axes[0,0]   # HERE!!! p={0:.4f}".format(p_value)
#ax.set(xlabel="Clinical MBF rest [mL/(min·g)]", ylabel="Low-Dose MBF rest [mL/(min·g)]") 
ax.set(xlabel="Clinical TPD stress [%]", ylabel="Low-Dose TPD stress [%]") 
plt.text(3, 22, "y={0:.2f}x+{1:.2f}".format(slope,intercept), horizontalalignment='left', size='medium', color='crimson', weight='semibold')
plt.text(3, 20, "R={0:.2f}".format(r_value), horizontalalignment='left', size='medium', color='crimson', weight='semibold')
plt.text(3, 18, "p=3.5937e-08", horizontalalignment='left', size='medium', color='crimson', weight='semibold')
plt.text(14, 7.5, "y={0:.2f}x{1:.2f}".format(slope2, intercept2), horizontalalignment='left', size='medium', color='steelblue', weight='semibold')
plt.text(14, 5.5, "R={0:.2f}".format(r_value2), horizontalalignment='left', size='medium', color='steelblue', weight='semibold')
plt.text(14, 3.5, "p=7.2873e-08", horizontalalignment='left', size='medium', color='steelblue', weight='semibold')
list1 = [i for i in range (1,25)]
plt.plot(list1, linewidth=2, linestyle='--', color = 'black')
plt.plot(list1)
plt.savefig('/homes/michellef/clinical_eval/TEST.png')
plt.close() """

# The horizontal plot is made using the hline function
my_range=range(1,len(df2.index)+1)

ordered_df = df2.sort_values(by='tpd_stress_100')

plt.hlines(y=my_range, xmin=ordered_df['tpd_stress_100'], xmax=ordered_df['tpd_stress_25'], color='grey', alpha=0.4)
plt.scatter(ordered_df['tpd_stress_100'], my_range, color='crimson', alpha=0.4 , label='100%')
plt.scatter(ordered_df['tpd_stress_25'], my_range, color='skyblue', alpha=1, label='25%')

plt.legend(loc='lower right')
plt.xlabel('TPD stress [%]')
plt.ylabel('Patient')

plt.axvline(x=0.0, color='g', linestyle='--', alpha = 0.4)
plt.axvline(x=5.0, color='g', linestyle='--', alpha = 0.4)
plt.text(0.8, 11.2, "Normal", horizontalalignment='left', size='medium', color='g', alpha = 0.5)
plt.axvline(x=5.0, color='orange', linestyle='--', alpha = 0.4)
plt.axvline(x=9.0, color='orange', linestyle='--', alpha = 0.4)
plt.text(5.8, 11.5, "Slight", horizontalalignment='left', size='medium', color='orange', alpha = 0.5)
plt.text(4.5, 11.0, "abnormality", horizontalalignment='left', size='medium', color='orange', alpha = 0.5)
plt.axvline(x=10.0, color='r', linestyle='--', alpha = 0.4)
plt.axvline(x=14.0, color='r', linestyle='--', alpha = 0.4)
plt.text(10.6, 11.5, "Moderate", horizontalalignment='left', size='medium', color='r', alpha = 0.5)
plt.text(10.0, 11.0, "abnormality", horizontalalignment='left', size='medium', color='r', alpha = 0.5) 
plt.axvline(x=15.0, color='brown', linestyle='--', alpha = 0.4)
plt.text(19.4, 11.5, "Severe", horizontalalignment='left', size='medium', color='brown', alpha = 0.5)
plt.text(18.6, 11.0, "abnormality", horizontalalignment='left', size='medium', color='brown', alpha = 0.5) 


""" plt.axvline(x=1.5, color='r', linestyle='--', alpha = 0.4)
plt.text(0.9, 11.0, "Elevated cardiac risk", horizontalalignment='left', size='medium', color='r', alpha = 0.5)
plt.axvline(x=2.3, color='g', linestyle='--', alpha = 0.4)
plt.text(2.4, 11.0, "Favourable prognosis", horizontalalignment='left', size='medium', color='g', alpha = 0.5) """
""" plt.text(22.5, 11.5, "Severely", horizontalalignment='left', size='medium', color='brown', alpha = 0.5)
plt.text(22.0, 11.0, "abnormal", horizontalalignment='left', size='medium', color='brown', alpha = 0.5)
plt.axvline(x=30.0, color='brown', linestyle='--', alpha = 0.4)
plt.axvline(x=30.0, color='r', linestyle='--', alpha = 0.4)
plt.text(30.5, 11.5, "Moderately", horizontalalignment='left', size='medium', color='r', alpha = 0.5)
plt.text(31.0, 11.0, "abnormal", horizontalalignment='left', size='medium', color='r', alpha = 0.5)
plt.axvline(x=39.0, color='r', linestyle='--', alpha = 0.4)
plt.axvline(x=39.0, color='orange', linestyle='--', alpha = 0.4)
plt.text(43.0, 11.5, "Mildly", horizontalalignment='left', size='medium', color='orange', alpha = 0.5)
plt.text(41.0, 11.0, "abnormal", horizontalalignment='left', size='medium', color='orange', alpha = 0.5)
plt.axvline(x=52.0, color='orange', linestyle='--', alpha = 0.4)
plt.axvline(x=52.0, color='g', linestyle='--', alpha = 0.4)
plt.axvline(x=70.0, color='g', linestyle='--', alpha = 0.4)
plt.text(55.0, 11.2, "Normal range", horizontalalignment='left', size='medium', color='g', alpha = 0.5)
plt.text(72.0, 11.2, "Hyperdynamic", horizontalalignment='left', size='medium', color='steelblue', alpha = 0.5)

plt.xlim(25,85) """

plt.savefig('/homes/michellef/clinical_eval/TEST2.png')
plt.close()