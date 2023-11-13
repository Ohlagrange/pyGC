#coding=utf-8
import glob
import os
import shutil
import pandas as pd
import numpy as np

#用于统计三维结果信息
output_path='./'
col_files=glob.glob(os.path.join(output_path, '*.col'))

output=pd.DataFrame()
for col_file in col_files:
    with open(col_file, 'r') as file:
        first_line = file.readline()
        CrankAngle = float(first_line.split()[0])
    Data = pd.read_csv(col_file,skiprows=1,delim_whitespace=True)
    #计算总体积
    total_volume=Data['volume'].sum()
    #温度大于2000K
    Temp_volume = Data[Data['TEMPERATURE'] > 2000]['volume'].sum()
    #当量比大于1
    EquivRatio_volume = Data[Data['EQUIV_RATIO'] > 0.5]['volume'].sum()
    #计算体积分数VolumeFration
    VF_Temp=Temp_volume/total_volume
    VF_EquivRatio=EquivRatio_volume/total_volume
    #输出
    output_CA=[CrankAngle,total_volume,Temp_volume,EquivRatio_volume,VF_Temp,VF_EquivRatio]
    row_to_add=pd.DataFrame([output_CA])
    output=output.append(row_to_add,ignore_index=True)   
    print(output_CA)
print(output)
    
