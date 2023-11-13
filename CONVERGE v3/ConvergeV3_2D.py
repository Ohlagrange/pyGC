import glob
import os
import shutil
import pandas as pd
import numpy as np

file_2D='input/340_performance_cal.py'
path='EVO_EVC'
dir_list=os.listdir(path)
performance=[]
performance=pd.DataFrame(performance)
for i in range(0,len(dir_list)):
    print(dir_list[i])
    shutil.copy(file_2D,os.path.join(path,dir_list[i]))
    os.chdir(os.path.join(path,dir_list[i]))
    os.system('python3.8 340_performance_cal.py')
    performance_out=pd.read_csv('performance_out.txt',sep='\t',usecols=[0],header=None)
    ming=pd.read_csv('performance_out.txt',sep='\t',usecols=[1],header=None)
    ming=ming.values.tolist()
    ming=list(map(list,zip(*ming)))
    performance_out.index=ming
    performance_out.columns=[dir_list[i]]
    performance=pd.concat([performance,performance_out],axis=1)
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
performance.to_csv('%s_performance_1.csv'%path)

