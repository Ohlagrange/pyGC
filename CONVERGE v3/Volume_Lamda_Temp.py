#coding=utf-8
import pandas as pd
import numpy as np

# 统计此刻有多少行数据
filename = 'B:\-c\output\disanzhong000087_+5.00353e+01.col'
d = open(filename)
line = d.readline()
count = 0
for index, line in enumerate(d):
    count += 1
count = count-1      # 统计.col文件里有多少行数据
print(count)



# 计算燃烧室总体积
f = pd.read_csv(filename,skiprows=2,delim_whitespace=True,header=None)
Volume = f.iloc[:,3]
Volume = np.array(Volume)
Volume = Volume.reshape(count,1)       # 列表转换为列矩阵

Total_Volume = 0
for i in range(count):
    Total_Volume += Volume[i,0]
print('Total_Volume:',Total_Volume)         # 输出所有网格体积

print("\n")

# 计算当量比大于某值的混合气分布体积
f = pd.read_csv(filename,skiprows=2,delim_whitespace=True,header=None)
Volume = f.iloc[:,3]
Volume = np.array(Volume)
Volume = Volume.reshape(count,1)       # 列表转换为列矩阵
Equiv_ratio = f.iloc[:,12]
Equiv_ratio = np.array(Equiv_ratio) 
Equiv_ratio = Equiv_ratio.reshape(count,1)  # 列表转换为列矩阵
L = np.dstack((Volume,Equiv_ratio))    # 将Volume和Equiv_ratio两列数据统合到三维矩阵C中



lamda_Volume_01 = 0
L_01 = L
for i in range(count):
    if 0.1 < L_01[i,0,1]:
        lamda_Volume_01 += L_01[i,0,0]
print('lamda_Volume_0.1:',lamda_Volume_01)         # 输出当量比大于0.1的所有网格体积

lamda_Volume_05 = 0
L_05 = L
for i in range(count):
    if 0.5 < L_05[i,0,1]:
        lamda_Volume_05 += L_05[i,0,0]
print('lamda_Volume_0.5:',lamda_Volume_05)         # 输出当量比大于0.5的所有网格体积

lamda_Volume_1 = 0
L_1 = L
for i in range(count):
    if 1 < L_1[i,0,1]:
        lamda_Volume_1 += L_1[i,0,0]
print('lamda_Volume_1:',lamda_Volume_1)         # 输出当量比大于1的所有网格体积

print("\n")

# 计算温度高于某值的高温区域分布体积
f = pd.read_csv(filename,skiprows=2,delim_whitespace=True,header=None)
Volume = f.iloc[:,3]
Volume = np.array(Volume)
Volume = Volume.reshape(count,1)       # 列表转换为列矩阵
Temp = f.iloc[:,11]
Temp = np.array(Temp) 
Temp = Temp.reshape(count,1)  # 列表转换为列矩阵
T = np.dstack((Volume,Temp))    # 将Volume和Temp两列数据统合到三维矩阵C中

Temp_Volume_1800 = 0
T_1800 = T
for i in range(count):
    if 1800 < T_1800[i,0,1]:
        Temp_Volume_1800 += T_1800[i,0,0]
print('Temp_Volume_1800:',Temp_Volume_1800)         # 输出温度大于1800K的所有网格体积

Temp_Volume_2000 = 0
T_2000 = T
for i in range(count):
    if 2000 < T_2000[i,0,1]:
        Temp_Volume_2000 += T_2000[i,0,0]
print('Temp_Volume_2000:',Temp_Volume_2000)         # 输出温度大于2000K的所有网格体积

Temp_Volume_2200 = 0
T_2200 = T
for i in range(count):
    if 2200 < T_2200[i,0,1]:
        Temp_Volume_2200 += T_2200[i,0,0]
print('Temp_Volume_2200:',Temp_Volume_2200)         # 输出温度大于2200K的所有网格体积