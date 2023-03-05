import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl


font = {'family' : 'Arial',
    #'weight' : 'medium',
    'size'   : 10,
    'style'  : 'normal'}


mpl.rcParams['xtick.direction'] = 'in'
mpl.rcParams['ytick.direction'] = 'in'
mpl.rc('font', **font)
fpt_types=["rdkfs","maccs","cicular4","cicular6","cicular8","circularfeat4","circularfeat6","circularfeat8"]

fig,axes=plt.subplots(ncols=3,nrows=3,figsize=(6, 3))
inds=[]
for i in range(3):
    for j in range(3):
        inds.append([i,j])
for i,tyype in enumerate(fpt_types):
    #load validation

    #load training
    axes[inds[i][0]][inds[i][1]].plot([0,1],[0,1],color='k')