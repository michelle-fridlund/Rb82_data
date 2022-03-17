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

patients = ['06fb290d-9666-47fb-a780-f796a9ca8e03_02',
            '0d64ef76-5f71-4485-b481-613f17beedfe_02',
            '3708dcc9-bce3-444d-b106-2909bf7a973b',
            '41309e5d-a63d-4150-b7c4-753f08143a3c',
            'a853b89f-e179-4a91-8f38-71298b5616d8',
            '19bf5adc-30df-44d9-a95d-9fd77b1a02ce',
            '6af03e0e-f2c7-483a-b617-b1563cfad550',
            '72a5cb53-6e9a-42d6-a6e5-827db88257aa',
            'b7b93d03-ad94-44cc-b581-3b07c9742c68',
            '1c9d3af1-f408-400c-b59e-583435fa1b9e']


s_edv_100 = [53,153,229,75,106,62,85,71,143,34]
s_edv_25 = [69,147,229,220,102,72,81,83,52,43]
s_edv_double = [62,155,218,72,100,55,81,169,141,40]
s_edv_single = [65,158,216,55,100,43,110,65,105,42]

s_esv_100 = [22,74,164,27,55,11,26,18,65,8]
s_esv_25 = [28,65,166,185,55,16,29,32,28,12]
s_esv_double = [22,68,150,23,54,9,29,116,61,9]
s_esv_single = [25,69,150,13,53,7,40,17,47,10]

s_ef_100 = [58,52,28,63,48,81,69,74,54,76]
s_ef_25 = [59,56,27,16,46,79,64,62,45,73]
s_ef_double = [64,56,31,67,46,84,64,31,57,77]
s_ef_single = [62,56,31,77,47,84,64,73,55,76]

r_edv_100 = [38,178,186,59,82,35,77,43,35,37]
r_edv_25 = [24,99,200,59,81,31,111,100,89,37]
r_edv_double = [194,105,203,56,68,51,111,41,87,33]
r_edv_single = [51,111,167,59,69,31,82,41,73,34]

r_esv_100 = [14,131,145,20,45,7,42,8,40,9]
r_esv_25 = [14,68,160,20,47,8,105,60,43,11]
r_esv_double = [132,65,153,19,41,13,105,8,41,8]
r_esv_single = [19,72,137,19,41,7,43,8,33,8]

r_ef_100 = [64,26,22,67,45,81,46,81,58,77]
r_ef_25 = [43,31,20,66,42,73,5,37,52,70]
r_ef_double = [32,38,25,66,40,75,5,81,53,74]
r_ef_single = [63,35,18,68,41,79,47,80,55,77]


df2 = pd.DataFrame({'s_ef_100': [58,52,28,63,48,81,69,74,54,76],
                    's_ef_25': [59,56,27,16,46,79,64,62,45,73],
                    's_ef_single': [62,56,31,77,47,84,64,73,55,76],},
                    index=['06fb290d-9666-47fb-a780-f796a9ca8e03_02',
                    '0d64ef76-5f71-4485-b481-613f17beedfe_02',
                    '3708dcc9-bce3-444d-b106-2909bf7a973b',
                    '41309e5d-a63d-4150-b7c4-753f08143a3c',
                    'a853b89f-e179-4a91-8f38-71298b5616d8',
                    '19bf5adc-30df-44d9-a95d-9fd77b1a02ce',
                    '6af03e0e-f2c7-483a-b617-b1563cfad550',
                    '72a5cb53-6e9a-42d6-a6e5-827db88257aa',
                    'b7b93d03-ad94-44cc-b581-3b07c9742c68',
                    '1c9d3af1-f408-400c-b59e-583435fa1b9e',])

df = pd.DataFrame(columns=['clinical','low','dose'])
for k,v in zip(r_ef_100,r_ef_25):
    df = df.append({'clinical': k, 'low':v, 'dose':'25%'},ignore_index=True)
for k,v in zip(r_ef_100,r_ef_single): # HERE!!!
    df = df.append({'clinical': k, 'low':v, 'dose':'25% (Denoised)'},ignore_index=True) # HERE!!!
df.clinical = df.clinical.astype('float')
df.low = df.low.astype('float')

# The horizontal plot is made using the hline function
my_range=range(1,len(df2.index)+1)

ordered_df = df2.sort_values(by='s_ef_100')

plt.hlines(y=my_range, xmin=ordered_df['s_ef_100'], xmax=ordered_df['s_ef_25'], color='grey', alpha=0.4)
plt.scatter(ordered_df['s_ef_100'], my_range, color='crimson', alpha=0.4 , label='100%')
plt.scatter(ordered_df['s_ef_25'], my_range, color='skyblue', alpha=1, label='25%')
plt.xlim(10,85)

plt.legend(loc='lower right')
plt.xlabel('LVEF stress [%]')
plt.ylabel('Patient')

plt.text(15.5, 11.5, "Severely", horizontalalignment='left', size='medium', color='brown', alpha = 0.5)
plt.text(15.0, 11.0, "abnormal", horizontalalignment='left', size='medium', color='brown', alpha = 0.5)
plt.axvline(x=30.0, color='brown', linestyle='--', alpha = 0.4)
plt.axvline(x=30.0, color='r', linestyle='--', alpha = 0.4)
plt.text(28.5, 11.5, "Moderately", horizontalalignment='left', size='medium', color='r', alpha = 0.5)
plt.text(29.0, 11.0, "abnormal", horizontalalignment='left', size='medium', color='r', alpha = 0.5)
plt.axvline(x=39.0, color='r', linestyle='--', alpha = 0.4)
plt.axvline(x=39.0, color='orange', linestyle='--', alpha = 0.4)
plt.text(43.0, 11.5, "Mildly", horizontalalignment='left', size='medium', color='orange', alpha = 0.5)
plt.text(41.0, 11.0, "abnormal", horizontalalignment='left', size='medium', color='orange', alpha = 0.5)
plt.axvline(x=52.0, color='orange', linestyle='--', alpha = 0.4)
plt.axvline(x=52.0, color='g', linestyle='--', alpha = 0.4)
plt.axvline(x=70.0, color='g', linestyle='--', alpha = 0.4)
plt.text(54.0, 11.2, "Normal range", horizontalalignment='left', size='medium', color='g', alpha = 0.5)
plt.text(70.0, 11.2, "Hyperdynamic", horizontalalignment='left', size='medium', color='steelblue', alpha = 0.5)

plt.savefig('/homes/michellef/clinical_eval/mar8_2022/lvef_stress.png')
plt.close()
""" #line_kws={'label':"y={0:.2f}x+{1:.2f}".format(slope,intercept)}    # HERE!!!
slope, intercept, r_value, p_value, std_err = stats.linregress(df2['r_ef_100'],df2['r_ef_25'])  
slope2, intercept2, r_value2, p_value2, std_err2 = stats.linregress(df2['r_ef_100'],df2['r_ef_single']) # HERE!!!
print(p_value, p_value2)

fgrid = sns.lmplot(x="clinical", y="low", data=df, hue = 'dose', palette="Set1")
ax = fgrid.axes[0,0]   # HERE!!! p={0:.4f}".format(p_value)
#ax.set(xlabel="Clinical MBF rest [mL/(min·g)]", ylabel="Low-Dose MBF rest [mL/(min·g)]") 
ax.set(xlabel="Clinical LVEF rest [%]", ylabel="Low-Dose LVEF rest [%]") 
plt.text(52, 25, "y={0:.2f}x+{1:.2f}".format(slope,intercept), horizontalalignment='left', size='medium', color='crimson', weight='semibold')
plt.text(52, 20, "R={0:.2f}".format(r_value), horizontalalignment='left', size='medium', color='crimson', weight='semibold')
plt.text(52, 15, "p={0:.4f}".format(p_value), horizontalalignment='left', size='medium', color='crimson', weight='semibold')
plt.text(35, 75, "y={0:.2f}x+{1:.2f}".format(slope2, intercept2), horizontalalignment='left', size='medium', color='steelblue', weight='semibold')
plt.text(35, 70, "R={0:.2f}".format(r_value2), horizontalalignment='left', size='medium', color='steelblue', weight='semibold')
plt.text(35, 65, "p= 2.4883e-07", horizontalalignment='left', size='medium', color='steelblue', weight='semibold')
#list1 = [i for i in range (1,80)]
#plt.plot(list1, linewidth=2, linestyle='--', color = 'black')
#plt.xlim(20,90)
#plt.ylim(20,100)
#plt.title('Random Gate + Static')
plt.savefig('/homes/michellef/clinical_eval/mar8_2022/ef_rest_single.png')
plt.close() """


""" plt.axvline(x=1.5, color='r', linestyle='--', alpha = 0.4)
plt.text(0.9, 11.0, "Elevated cardiac risk", horizontalalignment='left', size='medium', color='r', alpha = 0.5)
plt.axvline(x=2.3, color='g', linestyle='--', alpha = 0.4)
plt.text(2.4, 11.0, "Favourable prognosis", horizontalalignment='left', size='medium', color='g', alpha = 0.5)
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
plt.text(18.6, 11.0, "abnormality", horizontalalignment='left', size='medium', color='brown', alpha = 0.5) """


