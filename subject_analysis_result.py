import os
import math
import numpy as np
import matplotlib.pyplot as plt


sub_list = ['s1','s5','s7','s9','s11','s12','s16','s17','s19','s29','s31','s36','s39','s42','s43','s44','s45','s49','s53']
result = {}
result['r1']=[]
result['r2']=[]
result['r3']=[]
result['r4']=[]

for i in range(len(sub_list)):
    file_name = './result/sub_rank_analysis/record_source_analysis_{}.txt'.format(sub_list[i])
    result_file = open(file_name)
    count = 0
    sub_lines = []
    acc_lines = []
    for line in result_file.readlines():
        if count % 2 == 0:
            sub_lines.append(line)
        else:
            acc_lines.append(line)
        count += 1
    for ind, line in enumerate(acc_lines):
        result_acc = acc_lines[ind]
        result_str = result_acc.split(' ')
        score = float(result_str[1])
        result['r{}'.format(ind + 1)].append(score)


print(np.mean(result['r1']),np.mean(result['r2']),np.mean(result['r3']),np.mean(result['r4']))
print(np.std(result['r1'])/np.sqrt(len(sub_list)),np.std(result['r2'])/np.sqrt(len(sub_list)),np.std(result['r3'])/np.sqrt(len(sub_list)),np.std(result['r4'])/np.sqrt(len(sub_list)),)