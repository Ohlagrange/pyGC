from ruamel import yaml
import glob
import os
import shutil
import pandas as pd
import numpy as np
import multiprocessing as mp
from multiprocessing import Pool
import subprocess

surface_file='surface_changeH_repaired.dat'
output_file='surface.dat'
vertex=[]
triangle=[]
#读取文件
with open(surface_file,'r') as f:
    numbers=f.readlines()[0].split()
    numverts_tot=int(numbers[0])
    numverts=int(numbers[1])
    numtriangles=int(numbers[2])
    numseals=int(numbers[3])
with open(surface_file,'r') as f:
    for line in f.readlines()[1:numverts+1]:
        vertex.append(line.split())
with open(surface_file,'r') as f:
    for line in f.readlines()[numverts+1:numverts+numtriangles+1]:
        triangle.append(line.split())
with open(surface_file,'r') as f:
    seal=f.readlines()[numverts+numtriangles+1:]

#文件更改
deltaH=0.03  #boundary的z向移动距离
#写第一行
with open(output_file,'w') as f:
    first_line=" ".join([str(x) for x in numbers])
    f.write(first_line+'\n')
#获取需要调整的节点ID
def vertID(triangle,boundaryID):
    vertID=[]
    for s in triangle:
        if boundaryID in s[3]:
            vertID.append(s[0])
            vertID.append(s[1])
            vertID.append(s[2])
    vertID=set(vertID)
    return vertID

def change_boundaryID(oldID,newID,triangle):
    for s in triangle:
        if oldID in s[3]:
            s[3]=newID
    return triangle

vertID_saoqikou_ring=vertID(triangle,'18')
vertID_sleeve_top=vertID(triangle,'19')
vertID_20=vertID(triangle,'20')
#写入节点部分
with open(output_file,'a') as f:
    for s in vertex:
        if s[0] in vertID_saoqikou_ring:
            s[3]=str(float(s[3])+deltaH)
        elif s[0] in vertID_sleeve_top:
            s[3]=str(float(s[3])+deltaH)
        elif s[0] in vertID_20:
            s[3]=str(float(s[3])+deltaH)
        s=" ".join([str(x) for x in s])
        f.write(s+'\n')

#写入面网格部分
new_triangle=change_boundaryID('20','3',triangle)
with open(output_file,'a') as f:
    for s in new_triangle:
        s=" ".join([str(x) for x in s])
        f.write(s+'\n')

#写入seal部分
with open(output_file,'a') as f:
    for s in seal:
        f.write(s)
