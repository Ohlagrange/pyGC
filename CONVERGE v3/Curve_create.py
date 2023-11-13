#coding=utf-8
import os
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt


def curve_in_create(lift_data,file_name):

    lines = [
        "TEMPORAL",
        "valve",
        "direction  0 0 -1",
        "min_lift  1e-05",
        "CYCLIC         360.0",
        "crank          lift",
    ]

    # Open the file in write mode
    with open(file_name, "w") as file:
        # Write the first 6 lines
        for line in lines:
            file.write(line + "\n")

        # Write the lift_data
        for data in lift_data:
            line = f"{data[0]}     {data[1]}"
            file.write(line + "\n")

    print(f"File {file_name} created.")

def poly5(x,load,index):
    '''
    该函数应该包括8个5次多项式,load有四种：100Load，75Load，50Load，25Load，index有两种：qian，hou
    本文件中以75Load为基准，所有负荷采用相同的系数'''
    coefficients = {
        '100Load': {
            'qian': [-6.13325E-8, 3.37332E-6, -6.81692E-5, 6.6485E-4, -7.40193E-4, 4.92942E-4], 
            'hou': [-8.5746E-9, -7.19828E-7, -1.8133E-5, -1.1418E-4, -0.00117, -3.15943E-4], 
        },
        '75Load': {
            'qian': [-6.13325E-8, 3.37332E-6, -6.81692E-5, 6.6485E-4, -7.40193E-4, 4.92942E-4],
            'hou': [-8.5746E-9, -7.19828E-7, -1.8133E-5, -1.1418E-4, -0.00117, -3.15943E-4],
        },
        '50Load': {
            'qian': [-6.13325E-8, 3.37332E-6, -6.81692E-5, 6.6485E-4, -7.40193E-4, 4.92942E-4], 
            'hou': [-8.5746E-9, -7.19828E-7, -1.8133E-5, -1.1418E-4, -0.00117, -3.15943E-4],
        },
        '25Load': {
            'qian': [-6.13325E-8, 3.37332E-6, -6.81692E-5, 6.6485E-4, -7.40193E-4, 4.92942E-4], 
            'hou': [-8.5746E-9, -7.19828E-7, -1.8133E-5, -1.1418E-4, -0.00117, -3.15943E-4], 
        },
    }
    
    # Get the coefficients for the given load and index
    a, b, c, d, e, f = coefficients[load][index]

    # Evaluate the polynomial at x using the coefficients
    y = a * x**5 + b * x**4 + c * x**3 + d * x**2 + e * x + f
    
    return y

def EVlift(load,EVO,EVC,max_lift):
    '''
    该函数用于构造升程曲线'''
    CA=np.arange(0,360.5,0.5)
    S1=[]
    S2=[]
    for x in np.arange(0,180,0.5):
        if x<=EVO:
            S1.append(0)
        elif x>EVO and poly5(x-EVO,load,index='qian')<max_lift:
            S1.append(poly5(x-EVO,load,index='qian'))
        else:
            x1=x
            break
    for x in np.arange(x1,180,0.5):
        S1.append(max_lift)
    for x in np.arange(180,360.5,0.5)[::-1]:
        if x>=EVC:
            S2.append(0)
        elif x<EVC and poly5(x-EVC,load,index='hou')<max_lift:
            S2.append(poly5(x-EVC,load,index='hou'))
        else:
            x1=x+0.5
            break
    for x in np.arange(180,x1,0.5)[::-1]:
        S2.append(max_lift)
    S2.reverse()
    S=S1+S2
    EVlift=np.transpose([CA,S]).tolist()
    return EVlift

def create_sleeve_lift_in(SLO,SLC,v,SML):
    #SLO为滑块开启移动正时，SLC为关闭移动正时，v为滑块速度(m/CA)
    #SML为滑块最大位移
    delta=SML/v
    SL=[[0,0],
        [SLO,0],
        [SLO+delta,SML],
        [SLC,SML],
        [SLC+delta,0],
        [360,0]]

    lines = [
        "TEMPORAL",
        "valve",
        "direction  0 0 1",
        "min_lift  1e-05",
        "CYCLIC         360.0",
        "crank          lift",
    ]

    # Open the file in write mode
    with open('sleeve_lift.in', "w") as file:
        # Write the first 6 lines
        for line in lines:
            file.write(line + "\n")

        # Write the lift data
        for data in SL:
            file.write(f'{data[0]:.8f}   {data[1]:.8f}\n')  # format to 8 decimal places


# Function to read .in file and return lift data
def read_in_file(file_name):
    with open(file_name, "r") as file:
        lines = file.readlines()
    # Extract lift data which starts from the 7th line
    lift_data = [list(map(float, line.strip().split())) for line in lines[6:]]
    return lift_data

# Function to plot lift curves
def plot_lift_curves(*file_names):
    plt.figure(figsize=(10, 8))
    
    for file_name in file_names:
        lift_data = read_in_file(file_name)
        x_values = [data[0] for data in lift_data]
        y_values = [data[1] for data in lift_data]
        plt.plot(x_values, y_values, label=file_name)
        
    plt.xlabel('Crank Angle (degrees)')
    plt.ylabel('Lift (m)')
    plt.title('Lift Curves')
    plt.legend()
    #plt.grid(True)
    plt.show()

if __name__ == '__main__':
    
    #指定负荷，开闭正时及最大升程
    #原机数据 100Load: 110, 271; 75Load: 110, 265; 50Load: 113, 262; 25Load: 114, 256; 
    EVlift=EVlift('75Load',135,265,0.045)
    #指定文件名
    curve_in_create(EVlift,'EVLift_75Load.in')
    #plot_lift_curves('EVLift_75Load.in', 'EVLift_75Load_modi.in')

    create_sleeve_lift_in(165, 280, 0.005, 0.233)
