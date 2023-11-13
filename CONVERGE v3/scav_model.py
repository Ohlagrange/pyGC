import glob
import os
import shutil
import numpy as np
import pandas as pd


path='EVC_Pscan' 
dir_list=os.listdir(path)
d1=[]
d1=pd.DataFrame(d1)
d2=[]
d2=pd.DataFrame(d2)
scav_model=[]
scav_model=pd.DataFrame(scav_model) 
with pd.ExcelWriter('scav_model.xlsx') as writer:
    for i in range(0,len(dir_list)):
        print(dir_list[i])  
        os.chdir(os.path.join(path,dir_list[i]))
        passive_region=pd.read_csv('passive_region0.out',skiprows=5,delim_whitespace=True,header=None,usecols=[1,19])
        #passive_region0.out中只需要第1和第-1列的数据,算之前最好先确定一下
        region_flow=pd.read_csv('regions_flow.out',skiprows=5,delim_whitespace=True,header=None,usecols=[112,207])
        #regions_flow.out中只需要第112和第207列的数据
        CA=pd.read_csv('passive_region0.out',skiprows=5,delim_whitespace=True,header=None,usecols=[0])
        CA=CA.values.tolist()  
        CA=list(map(list,zip(*CA)))  
        passive_region.index=CA 
        region_flow.index=CA
        d1 = passive_region.iloc[:,0]/passive_region.iloc[:,1]  #d1=缸内已燃物质总质量/缸内总质量
        d1.index=CA
        d2 = region_flow.iloc[:,1]/region_flow.iloc[:,0]        #d2=排出气缸已燃物质的质量/排出气缸的总质量
        d2.index=CA
        scav_model=pd.concat([passive_region,region_flow,d1,d2],axis=1)
        scav_model.columns=['缸内已燃物质总质量', '缸内总质量', '排出气缸的总质量', '排出气缸已燃物质的质量','d1','d2']
        scav_model.to_excel(writer, sheet_name=str(dir_list[i])) #将每个case名命名为sheet名
        os.chdir(os.path.dirname(os.path.abspath(__file__))) #获取当前脚本的完整路径


