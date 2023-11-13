import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import re
import openpyxl

#data_plot dataframe gdx_plot.txt
def plot_value(CA,y):
    value_a=data_plot.at[int(CA)*2,y]
    value_c=data_plot.at[int(CA)*2+1,y]
    value_b=(value_c-value_a)/0.5*(CA-int(CA))+value_a
    return value_b

rootdir=r'K:\2020_340CFD\1D\340_noTC_simpleEV_Miller\100Load\EVC_Pscan'
list = os.listdir(rootdir)
########gdx输出######
Temp_TDC=[]
Temp_EVC=[]
Pres_EVC=[]
Mass_EVC=[]
O2_EVC=[]
CO2_EVC=[]
H2O_EVC=[]
Load=[]
Load=pd.DataFrame(Load)  #建立空的DataFrame
for i in range(0,len(list)):
    path = os.path.join(rootdir, list[i])
    path1=os.path.join(path,r'gdx.txt')
    path2=os.path.join(path,r'gdx_plot.txt')
    data=pd.read_csv(path1,sep='\t',usecols=[2])
    ming=pd.read_csv(path1,sep='\t',usecols=[0])
    data.index=ming
    Load=pd.concat([Load,data],axis=1,ignore_index=True)  #横向合并
    EVC_df=pd.DataFrame(data,index=pd.DataFrame(['Valve-Closing Timing Angle at 0 mm Lift;vtvc0:exhaust-1']))
    EVC=EVC_df.iloc[0,0]
    data_plot=pd.read_csv(path2,sep='\t',skip_blank_lines=True,skiprows=[1])
    Temp_TDC.append(plot_value(0,'Temperature:cyl-1'))
    Temp_EVC.append(plot_value(EVC,'Temperature:cyl-1'))
    Pres_EVC.append(plot_value(EVC,'Pressure:cyl-1'))
    Mass_EVC.append(plot_value(EVC,'Trapped Mass:cyl-1'))
    O2_EVC.append(plot_value(EVC,'Burned+Unburned O2, CO2, and H2O Mass Fractions:O2:cyl-1'))
    CO2_EVC.append(plot_value(EVC,'Burned+Unburned O2, CO2, and H2O Mass Fractions:CO2:cyl-1'))
    H2O_EVC.append(plot_value(EVC,'Burned+Unburned O2, CO2, and H2O Mass Fractions:H2O:cyl-1'))
    
    
#转速，inletP，inletT，扫气压力，排气阀关闭时刻，扫排气压比，压缩终点压力，循环喷油量，Tmax-unburned
#最高爆压，排气压力，排气集管温度，涡轮前温度，outletP，outletT，质量流量，捕集效率，过量空气系数，功率，油耗，NOx排放
results_name=['转速',
              'inletP',
              'inletT',
              '扫气压力',
              'EVO',
              'EVC',
              '扫排气压比',
              '压缩终点压力',
              '循环喷油量',
              '最高爆压',
              '排气压力',
              '排气集管温度',
              '涡轮前温度',
              'outletP',
              'outletT',
              '质量流量',
              '捕集效率',
              '过量空气系数',
              '功率',
              '油耗',
              'NOx']
col=['Engine Speed (cycle average);avgrpm:engine-1',
     'Average Pressure;pav:inlet-1',
     'Average Temperature;tav:inlet-1',
     'Average Pressure;pav:plenum_scan-1',
     'Valve-Open Timing Angle at 0 mm Lift;vtvo0:exhaust-1',
     'Valve-Closing Timing Angle at 0 mm Lift;vtvc0:exhaust-1',
     'diff:PARAMETER',
     'Maximum Pressure, Motoring;pmaxmot:cyl-1',
     'Injected Mass per Cycle;injrat:Injector-1',
     'Maximum Pressure During Combustion;pmaxcomb:cyl-1',
     'Average Pressure;pav:plenum_exh-1',
     'Average Temperature;tav0:plenum_exh-1',
     'Mass Averaged Temperature (Inlet);tavl:exhaPlenum_turb02-1',
     'ex_p:PARAMETER',
     'ex_t:PARAMETER',
     'Average Mass Flow Rate (Inlet);favl:WMC',
     'Trapping Ratio;trappc:cyl-1',
     'Effective Lambda at EVO;efflambda:cyl-1',
     'Brake Power;bkw:engine-1',
     'BSFC - Brake Specific Fuel Consumption, Cyl;bsfc:engine-1',
     'Brake Specific NOx - Cylinder Out;bsnoxnew:engine-1']

#列名转换格式
col_df=pd.DataFrame(col)
#
Load_1=pd.DataFrame(Load.values.T,index=Load.columns,columns=Load.index)#转置
Load_2=pd.DataFrame(Load_1,columns=col_df)
Load_2.columns=results_name
plot_v={'Temp_TDC':Temp_TDC,
        'Temp_EVC':Temp_EVC,
        'Pres_EVC':Pres_EVC,
        'Mass_EVC':Mass_EVC,
        'O2_EVC':O2_EVC,
        'CO2_EVC':CO2_EVC,
        'H2O_EVC':H2O_EVC}
plot=pd.DataFrame(plot_v)
Load_3=pd.concat([Load_2,plot],axis=1)
#
writer=pd.ExcelWriter('Results.xlsx')
Load_3.to_excel(writer,'Sheet1')
writer.save()

