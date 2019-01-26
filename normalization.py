#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 10 16:48:41 2019

@author: nizi
"""
import scipy.io as scio
import os
import numpy as np

path1 = r'E:/Yue/Entire Data/CNMC/hospital_data/L_Original/'
path2 = r'E:/Yue/Entire Data/CNMC/hospital_data/R_Original/'
resultpath1 = r'E:/Yue/Entire Data/CNMC/hospital_data/L/'
resultpath2 = r'E:/Yue/Entire Data/CNMC/hospital_data/R/'

def findmat(path):
    files = []
    pathlist = os.listdir(path)
    pathlist.sort()
    for file in pathlist:
        files.append(file)
    return files

def findboundary(path):
    Max = -65535
    Min = 65535
    files = findmat(path)
    for file in files:
        data = scio.loadmat(path+file)
        a = np.array(data['z1'])
        if(a.max() > Max):
            Max = a.max()
        if(a.min() < Min):
            Min = a.min()
    return Max,Min

def normalization(path,respath,Max,Min):
    files = findmat(path)
    for file in files:
        print(file)
        data = scio.loadmat(path + file)
        a = np.array(data['z1'])
        b = (a-Min)/(Max-Min)
        scio.savemat(respath+file,{'z1': b})

Lmax, Lmin = findboundary(path1)
Rmax, Rmin = findboundary(path2)

#print(Lmax,Lmin)
#print(Rmax,Rmin)
#maxvalue = max(Lmax, Rmax)
#minvalue = min(Lmin, Rmin)

if(Lmax > Rmax):
    maxvalue = Lmax
else:
    maxvalue = Rmax
if(Lmin < Rmin):
    minvalue = Lmin
else:
    minvalue = Rmin

print(maxvalue, minvalue)

normalization(path1,resultpath1,maxvalue,minvalue)
normalization(path2,resultpath2,maxvalue,minvalue)