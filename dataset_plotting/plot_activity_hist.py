import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
from multiprocessing import Pool
import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"    
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import itertools
from time import time
import random
from matplotlib.animation import FuncAnimation
import spektral
from spektral.layers.convolutional.gcn_conv import GCNConv
import sys
import shutil

recalc=1
if recalc:

    o_dir="good_n_bad/"
    dataset="/home/jacob/Analysis/GPCR/nnet_lig/interactions_total.tsv"
    data={}
    data["InChI"]=[]
    data["UniProt"]=[]
    data["exp"]=[]
    data["val"]=[]
    fo=open(dataset,'r')
    i=-1
    count=0
    for line in fo:
        #print(i)
        if (i!=-1) and len(line)>0:
            linet=line.strip().split('\t')
            dont_save=0
            if linet[4]!="nM":
                dont_save=1
            if linet[2] not in ["IC50","Ki","EC50"]:
                dont_save=1
            if dont_save==0:
                try:
                    flott=float(linet[3])
                    if flott <=0:
                        dont_save=1
                    else:
                        data["val"].append(float(linet[3]))
                except:
                    dont_save=1
                    #try:
                    #   data["val"].append(float(linet[3].split("<")[1]))
                    #except:
                    #   try:
                    #       data["val"].append(float(linet[3].split(">")[1]))
                    #       #print(0.5*(float(line[3].split("-")[0])+float(line[3].split("-")[1])))
                    #   except:
                    #       try:
                    #           data["val"].append(0.5*(float(linet[3].split("-")[0])+float(linet[3].split("-")[1])))
                    #           #print(data["val"][-1])
                    #       except:
                    #           
                    #           # else:
                    #           #   print("error:",linet[3])
                    #           #   print(line)
                    #           #   data["val"].append(linet[3])
                # if linet[1]=="P":
                #   print(line)
                #   print(linet[1])
            if dont_save==0:
                data["UniProt"].append(linet[0])
                data["InChI"].append(linet[1])
                data["exp"].append(linet[2])
        i+=1
    fo.close()


data=np.log10(np.array(data["val"])/1000)#log10(micromolar)


act_dat=[]
for i in range(63):
    dat=np.load('../data_sets/GLASS_6_2_21mol_act_'+str(i*10000)+'.npy')
    act_dat.append(dat)
data=np.concatenate(act_dat,axis=0)


data=np.log10(np.array(data)/1000)#log10(micromolar)
data=data[data>-5]
data=data[data<5]
print(np.sum(data<10),np.sum(data>=0))
fig,axes=plt.subplots(figsize=(3,3))
plt.hist(data,bins=50,color='blue')
plt.xlabel('log(Activity / '+r'$\mu$'+'M)')
plt.ylabel('Number of Compounds')
plt.xticks([-5,-3,-1,1,3,5])
plt.plot([0,0],[0,31000],color='k')
plt.tight_layout()
plt.savefig('figures/Fig_activity_histogram.png',dpi=300,transparent=True)
plt.show()