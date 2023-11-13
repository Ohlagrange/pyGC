import glob
import os
import shutil
import pandas as pd
import numpy as np

file_2D='340_performance_cal_2310.py'
out_name='340_performance_output.txt'
path='./'
dir_list=[f for f in os.listdir(path) if os.path.isdir(os.path.join(path, f))]
performance=[]
performance=pd.DataFrame(performance)
for i in range(0,len(dir_list)):
    if 'converge.done' in os.listdir(dir_list[i]):
        #print(dir_list[i])
        if '340_performance_output.txt' in os.listdir(dir_list[i]):
            shutil.copy(file_2D,os.path.join(path,dir_list[i]))
            os.chdir(os.path.join(path,dir_list[i]))
            os.system('python3.8 %s'%file_2D) 
        else:
            print(dir_list[i])
            shutil.copy(file_2D,os.path.join(path,dir_list[i]))
            os.chdir(os.path.join(path,dir_list[i]))        
            os.system('python3.8 %s'%file_2D)
        performance_out=pd.read_csv(out_name,sep='\t',usecols=[0],header=None)
        ming=pd.read_csv(out_name,sep='\t',usecols=[1],header=None)
        ming=ming.values.tolist()
        ming=list(map(list,zip(*ming)))
        performance_out.index=ming
        performance_out.columns=[dir_list[i]]
        performance=pd.concat([performance,performance_out],axis=1)
        os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Transpose the DataFrame
performance_transposed = performance.transpose()

# Sort the DataFrame by row names (index)
performance_transposed_sorted = performance_transposed.sort_index(key=lambda x: x.str.lower())

# Output the sorted, transposed DataFrame to an Excel file
output_path = '340_performance_results_2310.xlsx'
performance_transposed_sorted.to_excel(output_path)