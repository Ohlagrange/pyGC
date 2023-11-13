import glob
import os
import shutil
import multiprocessing as mp
from multiprocessing import Pool

#指定目录增加文件
file=r'K:\2020_340CFD\1D\340_noTC_simpleEV_Miller\gdx_plot.exp'
rootdir=r'K:\2020_340CFD\1D\340_noTC_simpleEV_Miller\50Load_1\EVC_Pscan'
list = os.listdir(rootdir)
def job(x):
    path = os.path.join(rootdir, list[x])
    shutil.copy(file,path)
    os.chdir(path)
    os.system(r'gtexport gdx_plot.exp')

if __name__=='__main__':
    pool=Pool(28)
    for i in range(0,len(list)):
        pool.apply_async(job,(i,))
    pool.close()
    pool.join()
