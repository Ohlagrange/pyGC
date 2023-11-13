from ruamel import yaml
import glob
import os
import shutil
import multiprocessing as mp
from multiprocessing import Pool

#修改spray.in
def change_spray_in(output_path,SOI_0,SOI_1):
    spray_in=os.path.join(input_path, 'spray.in')
    with open(spray_in, encoding="utf-8") as f:
        content = yaml.load_all(f, Loader=yaml.RoundTripLoader)
        data=list(content)
        # 修改spray文件中的参数
        data[1]['injectors'][0]['injector']['injector_control']['start_time']=SOI_0
        data[1]['injectors'][1]['injector']['injector_control']['start_time']=SOI_1
    output_file = os.path.join(output_path, 'spray.in')
    with open(output_file, 'w', encoding="utf-8") as nf:
        yaml.dump_all(data, nf, Dumper=yaml.RoundTripDumper,default_flow_style=False,allow_unicode=True,indent=4,block_seq_indent=3)

#修改initial.in
def change_initial(output_path,mf_O2,mf_CO2,mf_H2O):
    initial_in=os.path.join(input_path, 'initial.in')
    with open(initial_in, encoding="utf-8") as f:
        content = yaml.load_all(f, Loader=yaml.RoundTripLoader)
        data=list(content)
        #修改initial文件中的参数
        data[0][1]['region']['species_init']['O2']=mf_O2
        data[0][1]['region']['species_init']['CO2']=mf_CO2
        data[0][1]['region']['species_init']['H2O']=mf_H2O
        data[0][1]['region']['species_init']['N2']=1-mf_CO2-mf_H2O-mf_O2
    output_file = os.path.join(output_path, 'initial.in')
    with open(output_file, 'w', encoding="utf-8") as nf:
        yaml.dump_all(data, nf, Dumper=yaml.RoundTripDumper,default_flow_style=False,allow_unicode=True,indent=4,block_seq_indent=3)

def job(x):
    #读取DOE数据
    index=x
    DOE = DOE_list[index]
    #创建子文件夹及复制文件
    output_path = r'Premix_Combustion\SOI0_%s_SOI1_%s' %(DOE[0],DOE[1])
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    for file in os.listdir(input_path):
        shutil.copyfile(os.path.join(input_path, file), os.path.join(output_path, file))
    #修改文件
    change_spray_in(output_path,float(DOE[0]),float(DOE[1]))
    
if __name__ == '__main__':
    input_path = 'input'
    DOE_file = 'DOE.txt'
    DOE_list = []
    with open(DOE_file, 'r') as f:
        for line in f.readlines()[1:]:
            DOE_list.append(line.split())
    job(0)
    job(1)
    

