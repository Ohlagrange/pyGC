#!/usr/bin/python3
# -- coding:utf8 --

fname=r"F:\340_mesh_cali\base_spray_mesh\restart0053.rst"

import h5py
import numpy as np

def h5list(f,tab):
    print(tab,'Group:',f.name,'len:%d'%len(f))
    mysp2=tab[:-1]+ '  |-*'
    for vv in f.attrs.keys():  # 打印属性
        print(mysp2,end=' ')
        print('%s = %s'% (vv,f.attrs[vv]))
    mysp=tab[:-1] + '  |-'
    for k in f.keys():
        d = f[k]
        if isinstance(d,h5py.Group):
            h5list(d,mysp)
        elif isinstance(d,h5py.Dataset):
            print(mysp,'Dataset:',d.name,'(size:%d)'%d.size)
            mysp1=mysp[:-1]+ '  |-'
            print(mysp1,'(dtype=%s)'%d.dtype)
            if d.dtype.names is not None:
                print(mysp,end=' ')
                for vv in d.dtype.names:
                    print(vv,end=',')
                print()
            mysp2=mysp1[:-1]+ '  |-*'
            for vv in d.attrs.keys():  # 打印属性
                print(mysp2,end=' ')
                try:
                    print('%s = %s'% (vv,d.attrs[vv]))
                except TypeError as e:
                    print('%s = %s'% (vv,e))
                except:
                    print('%s = ?? Other ERR'% (vv,))
            #print(d[:12])  # 打印12组数据看看
        else:
            print('??->',d,'Unkown Object!')

def time_step(f):
    print('TIME_STEP = %s' %f.attrs['TIME_STEP'])

f = h5py.File(fname,'r')
#print(fname)
#所有属性只能print一次，第二次会报错
#h5list(f,'')
time_step(f)
f.close()
