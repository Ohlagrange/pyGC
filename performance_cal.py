#coding=utf-8
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from ruamel import yaml
from ruamel.yaml.compat import StringIO
import os
from collections import Counter

'''
t=2 or 4
从工作目录读取数据：
工作目录下有converge.done文件则开始统计，否则返回计算未完成
读取initial.in中的区域分块，若区域数量多于1个，则在读取out文件时加后缀region0（缸内区域）
读取input.in中的开始计算时刻及结束时刻
读取in文件中的数据，如SOI，fuelmass，
读取out文件中的曲线，分为全部和指定区间的曲线
读取out文件中特定曲轴转角下的值
读取boundary.in中的名称，编号及区域，统计缸内区域的传热量
内燃机原理相关计算 
指定冲程t
指定燃料热值
'''
def check_cycle(t):
    with open('inputs.in',encoding="utf-8") as f:
        content = yaml.load_all(f,Loader=yaml.RoundTripLoader)
        inputs=list(content)
        CA_From=inputs[1]['simulation_control']['start_time']
        CA_End=inputs[1]['simulation_control']['end_time']

    #根据冲程数调整
    seq = [i*360*(t/2) for i in range(10)]

    # 对列表中的每个元素进行检查，看它是否位于(CA_From, CA_End)区间内
    for i in range(len(seq)):
        if CA_From <= seq[i] < CA_End:
            cycle = i
            break

    return CA_From, CA_End, cycle    

def engine_in(para_name):
    with open('engine.in',encoding="utf-8") as f:
        content = yaml.load_all(f,Loader=yaml.RoundTripLoader)
        engine=list(content)
        para_value=engine[1][para_name]
    return para_value

def boundary_info(boundary_name, info_name):
    with open('boundary.in', encoding="utf-8") as f:
        content = yaml.load_all(f, Loader=yaml.RoundTripLoader)
        boundary_data = list(content)
    boundary_conditions = boundary_data[1]['boundary_conditions']

    # Search for the specified boundary
    for condition in boundary_conditions:
        boundary = condition['boundary']
        if boundary['name'] == boundary_name:
            # If the boundary is found, search for the specified info
            return boundary.get(info_name, None)
    
    # If the boundary is not found, return None
    return None


def initial_in():
    with open('initialize.in', encoding="utf-8") as f:
        content = yaml.load_all(f,Loader=yaml.RoundTripLoader)
        initial_in=list(content)
        region_count=len(initial_in[1])
        
    return {'region_count': region_count}

def spray_in():
    fuelmass = 0
    with open('spray.in', encoding="utf-8") as f:
        content = yaml.load_all(f, Loader=yaml.RoundTripLoader)
        spray_in=list(content)
        injector_number=len(spray_in[1]['injectors'])
        for i in range(0,injector_number,1):
            fuelmass=fuelmass+spray_in[1]['injectors'][i]['injector']['injector_control']['tot_mass']
    #读取的结果
    fuelmass=fuelmass*1000 #g
    SOI=spray_in[1]['injectors'][0]['injector']['injector_control']['start_time'] #第一个喷油器的正时

    return {'fuelmass': fuelmass, 'SOI': SOI}

#对列名进行修正，若有重名，则自动重命名
def process_list(lst):
    count_dict = {}
    result = []
    for item in lst:
        if item in count_dict:
            count_dict[item] += 1
            result.append(f"{item}_{count_dict[item]}")
        else:
            count_dict[item] = 0
            result.append(item)
    return result

def read_outfile(file_name):
    if os.path.isfile(file_name):
    # Read the third row for column names
        with open(file_name, 'r') as f:
            lines = [next(f) for x in range(5)]  # get the first 5 lines
            third_line = lines[2].strip().split()[1:]  # third line, skip the first element '#'
            fifth_line = lines[4].strip().split()[1:]  # fifth line, skip the first element '#'
        if len(fifth_line) <= 1:
        # 使用第三行作为列名
            columns = third_line
        else:
            # 合并第三行和第五行作为列名
            columns = [f"{col} [{desc}]" if desc and desc != "(none)" else col for col, desc in zip(third_line, fifth_line)]
            # Skip the first 5 rows of metadata, and use whitespace as delimiter

        columns=process_list(columns)
        data = pd.read_csv(file_name, skiprows=5, delim_whitespace=True, names=columns)
    return data

def CA_getY(data, CA, Y_name, cycle):

    Target_Y = np.interp(CA+360*(cycle-1), data['Crank'], data[Y_name])

    return Target_Y

def store(results, key, value):
    results[key] = value
    return value


class Performance:
    def __init__(self):
        pass

    def calculate(self):

        #前置输入
        C7H16_LHV = 44.9143  #MJ/kg
        t = 2

        # 读取in文件
        Bore = engine_in('bore')
        Stroke = engine_in('stroke')
        self.rpm = engine_in('rpm')
        spray_in_values = spray_in()
        fuelmass = spray_in_values['fuelmass']
        SOI = spray_in_values['SOI']
        self.CA_From, self.CA_End, self.cycle = check_cycle(t)

        # 读取out文件
        emissions = read_outfile('emissions_region0.out')
        NOx = emissions['NOx'].iloc[-1]
        Soot = emissions['Hiroy_Soot'].iloc[-1]
        HC = emissions['HC'].iloc[-1]
        CO = emissions['CO'].iloc[-1]

        thermo=read_outfile('thermo_region0.out')
        CrankAngle=thermo['Crank']
        Pressure=thermo['Pressure']
        Volume=thermo['Volume']
        Temp=thermo['Mean_Temp']
        HeatRealeaseRate=thermo['HR_Rate']
        IntegratedHeatRealease=thermo['Integrated_HR']

        self.Work=np.trapz(Pressure,Volume)*1000000
        self.IMEP=self.Work/(np.pi*(Bore/2)**2*Stroke)/100000
        self.IndicatedPower=self.IMEP*(np.pi*(Bore/2)**2*Stroke)*self.rpm/(30*t)*100
        self.ISFC=fuelmass/(self.Work/3600000)
        self.INOx=NOx*1000/(self.Work/3600000)

        #region_flow改为全输出后，这个值为负数，所以添加负号
        region_flow=read_outfile('regions_flow.out')
        self.Total_mass_regions_0_to_1 = -CA_getY(region_flow, 300.0, 'Tot_Mass [Regions_0_to_1]', self.cycle)

        passive=read_outfile('passive_region0.out')
        self.INTAKE_region0_remain = CA_getY(passive, 300.0, 'INTAKE', self.cycle)

        self.Total_mass_region0_EVC = CA_getY(thermo, 300, 'Mass', self.cycle)

        dynamic=read_outfile('dynamic_region0.out')
        self.swirl_TDC = CA_getY(dynamic, 360.0, 'Swirl_Ratio', self.cycle)

        self.PeakFiringPressure=Pressure.max()
        self.PFP_CA=CrankAngle.iloc[Pressure.idxmax()]-(self.cycle-1)*360*(t/2)
        self.PeakIncylinerTemp=Temp.max()
        PressureRiseRate=np.diff(Pressure)/np.diff(CrankAngle)
        self.PeakPRR=PressureRiseRate.max()
        #PeakPRR_CA=CrankAngle.iloc[np.argmax(PressureRiseRate)]
        self.HRR_Peak=HeatRealeaseRate.max()
        self.HRR_Peak_CA=CrankAngle.iloc[HeatRealeaseRate.idxmax()]-(self.cycle-1)*360*(t/2)

        self.ISoot=Soot*1000/(self.Work/3600000)
        self.IHC=HC*1000/(self.Work/3600000)
        self.ICO=CO*1000/(self.Work/3600000)

        TotalHeatRealease=IntegratedHeatRealease.iloc[-1]
        CA_10_abs=abs(IntegratedHeatRealease[:]-(TotalHeatRealease*0.1))
        self.CA10=CrankAngle.iloc[CA_10_abs.idxmin()]-(self.cycle-1)*360*(t/2)
        CA_50_abs=abs(IntegratedHeatRealease[:]-(TotalHeatRealease*0.5))
        self.CA50=CrankAngle.iloc[CA_50_abs.idxmin()]-(self.cycle-1)*360*(t/2)
        CA_90_abs=abs(IntegratedHeatRealease[:]-(TotalHeatRealease*0.9))
        self.CA90=CrankAngle.iloc[CA_90_abs.idxmin()]-(self.cycle-1)*360*(t/2)
        self.IgnitionDelay=self.CA10-SOI
        self.CombustionDuration=self.CA90-self.CA10

        species=read_outfile('species_mass_region0.out')
        mass_O2=CA_getY(species,300,'O2', self.cycle)
        self.lambda_region0=mass_O2/0.233/(fuelmass/1000)/14.7

        #能量平衡分析
        self.TotalEnergy=C7H16_LHV*fuelmass*1000
        self.IncompletedCombustion=self.TotalEnergy-TotalHeatRealease

        head_id=boundary_info('head','id')
        self.HT_head=read_outfile('bound%s-wall.out'%head_id)['Tot_HT_xfer'].iloc[-1]

        piston_id=boundary_info('piston','id')
        self.HT_piston=read_outfile('bound%s-wall.out'%piston_id)['Tot_HT_xfer'].iloc[-1]

        liner_id=boundary_info('liner','id')
        self.HT_liner=read_outfile('bound%s-wall.out'%liner_id)['Tot_HT_xfer'].iloc[-1]

        valvebottom_id=boundary_info('valvebottom','id')
        self.HT_valvebottom=read_outfile('bound%s-wall.out'%valvebottom_id)['Tot_HT_xfer'].iloc[-1]

        self.HT_total = self.HT_head + self.HT_liner + self.HT_piston + self.HT_valvebottom

        self.ExhaustEnergy=self.TotalEnergy-self.IncompletedCombustion-self.HT_total-self.Work

        self.EnergyProportion_Work=self.Work/self.TotalEnergy
        self.EnergyProportion_IncompletedCombustion=self.IncompletedCombustion/self.TotalEnergy
        self.EnergyProportion_HT_total=self.HT_total/self.TotalEnergy
        self.EnergyProportion_Exhaust=self.ExhaustEnergy/self.TotalEnergy

        return self  # 返回self以方便访问类的属性


if __name__ == '__main__' :

    performance = Performance().calculate()
    with open('340_performance_output.txt', 'w') as f:
        for key, value in vars(performance).items():
            f.write(f'{value}\t {key}\n')



