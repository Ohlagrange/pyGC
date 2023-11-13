import glob
import os
import shutil
import multiprocessing as mp
from multiprocessing import Pool

# 读取文件信息
import subprocess

input_path = 'input'
gtm_file = glob.glob(os.path.join(input_path, '*.gtm'))[0]
with open(gtm_file, 'r') as f:
    lines = f.readlines()

DOE_file = 'DOE.txt'
DOE_list = []
with open(DOE_file, 'r') as f:
    for line in f.readlines()[1:]:
        DOE_list.append(line.split())

# 指定路径（两处文件路径），<entry>EVC</entry>
def job(x):
    index = x
    DOE = DOE_list[index]
    output_path = r'EVC_Pscan\EVC_%s_Pscan_%s' %(DOE[0],DOE[1])
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    for file in os.listdir(input_path):
        shutil.copyfile(os.path.join(input_path, file), os.path.join(output_path, file))
    output_file = os.path.join(output_path, os.path.basename(gtm_file))
    with open(output_file, 'w') as f:
        for s in lines:
            if '<entry>EVC</entry>' in s:
                s = s.replace('<entry>EVC</entry>', DOE[0])
            elif '<entry>Pscan</entry>' in s:
                s=s.replace('<entry>Pscan</entry>', DOE[1])
            f.write(s)

    # 运行 0_run.bat
    subprocess.call(r'EVC_Pscan\EVC_%s_Pscan_%s\0_run.bat'  %(DOE[0],DOE[1]), shell=True)

if __name__=='__main__':
    pool=Pool(28)
    for i in range(0,len(DOE_list)):
        pool.apply_async(job,(i,))
    pool.close()
    pool.join()
