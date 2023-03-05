import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

def get_auc_trap(x_pts,y_pts):
    auc=0.
    n_pts=x_pts.shape[0]
    for i in range(n_pts-1):
        width=np.abs(x_pts[i+1]-x_pts[i])
        height=0.5*(y_pts[i+1]+y_pts[i])
        auc+=width*height
    return auc

font = {'family' : 'Arial',
    #'weight' : 'medium',
    'size'   : 9,
    'style'  : 'normal'}


mpl.rcParams['xtick.direction'] = 'in'
mpl.rcParams['ytick.direction'] = 'in'
mpl.rc('font', **font)

#21 dilation=[1,3,5]
# if mod_i==22:
#     dilations=[1,3]#[]#
# elif mod_i==23:
#     dilations=[1]#[]#
# elif mod_i==24:
#     dilations=[]#[]#

loc_0=1.
bw=.25
angle=90

plt.figure(figsize=[1.5,1.5])

dat=np.load("../roc_compare/roc_"+str(21)+"_dat.npy")
auc=get_auc_trap(dat[0],dat[1])
dat=np.load("../roc_compare/roc_"+str(50)+"_dat.npy")
auc1=get_auc_trap(dat[0],dat[1])
dat=np.load("../roc_compare/roc_"+str(60)+"_dat.npy")
auc2=get_auc_trap(dat[0],dat[1])
aucs=np.array([auc,auc1,auc2])

color='b'
plt.bar(loc_0,np.average(aucs),yerr=np.std(aucs),width=bw*.9,label="d = [1,2,4,6]",color=color)
plt.text(loc_0+.13,np.average(aucs)+.005,str(np.average(aucs))[0:6], rotation=angle, rotation_mode='anchor')
loc_0+=bw

dat=np.load("../roc_compare/roc_"+str(22)+"_dat.npy")
auc=get_auc_trap(dat[0],dat[1])
dat=np.load("../roc_compare/roc_"+str(51)+"_dat.npy")
auc1=get_auc_trap(dat[0],dat[1])
dat=np.load("../roc_compare/roc_"+str(61)+"_dat.npy")
auc2=get_auc_trap(dat[0],dat[1])
aucs=np.array([auc,auc1,auc2])
color='g'

plt.bar(loc_0,np.average(aucs),yerr=np.std(aucs),width=bw*.9,label="d = [1,2,4]",color=color)
plt.text(loc_0+.13,np.average(aucs)+.005,str(np.average(aucs))[0:6], rotation=angle, rotation_mode='anchor')
loc_0+=bw


dat=np.load("../roc_compare/roc_"+str(23)+"_dat.npy")
auc=get_auc_trap(dat[0],dat[1])
dat=np.load("../roc_compare/roc_"+str(52)+"_dat.npy")
auc1=get_auc_trap(dat[0],dat[1])
dat=np.load("../roc_compare/roc_"+str(62)+"_dat.npy")
auc2=get_auc_trap(dat[0],dat[1])
aucs=np.array([auc,auc1,auc2])
color='r'

plt.bar(loc_0,np.average(aucs),yerr=np.std(aucs),width=bw*.9,label="d = [1,2]",color=color)
plt.text(loc_0+.13,np.average(aucs)+.005,str(np.average(aucs))[0:6], rotation=angle, rotation_mode='anchor')

loc_0+=bw

dat=np.load("../roc_compare/roc_"+str(24)+"_dat.npy")
auc=get_auc_trap(dat[0],dat[1])
dat=np.load("../roc_compare/roc_"+str(53)+"_dat.npy")
auc1=get_auc_trap(dat[0],dat[1])
dat=np.load("../roc_compare/roc_"+str(63)+"_dat.npy")
auc2=get_auc_trap(dat[0],dat[1])
aucs=np.array([auc,auc1,auc2])
color='orange'

plt.bar(loc_0,np.average(aucs),yerr=np.std(aucs),width=bw*.9,label="d = [1]",color=color)
plt.text(loc_0+.13,np.average(aucs)+.005,str(np.average(aucs))[0:6], rotation=angle, rotation_mode='anchor')


plt.ylim([.80,1.])
plt.ylabel("AUC")
plt.xticks([])
#plt.legend()

ax=plt.gca()
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
plt.savefig("figures/Figure_dilation_training_curves_inset.png",dpi=300,transparent=True)
plt.show()
exit()


plt.figure(figsize=[3,3])

mod_i=21
color='b'
train=[]
fo=open("training_curves/mod_"+str(mod_i)+"_train.csv",'r')
i=0
for line in fo:
    if i > 0:
        train.append(float(line.strip().split(",")[2]))
    i+=1
fo.close()
valid=[]
fo=open("training_curves/mod_"+str(mod_i)+"_valid.csv",'r')
i=0
for line in fo:
    if i > 0:
        valid.append(float(line.strip().split(",")[2]))
    i+=1
fo.close()
plt.plot(train,label="d = [1,2,4,6]",color=color)
plt.plot(valid,color=color,ls='--')

mod_i=22
color='g'
train=[]
fo=open("training_curves/mod_"+str(mod_i)+"_train.csv",'r')
i=0
for line in fo:
    if i > 0:
        train.append(float(line.strip().split(",")[2]))
    i+=1
fo.close()
valid=[]
fo=open("training_curves/mod_"+str(mod_i)+"_valid.csv",'r')
i=0
for line in fo:
    if i > 0:
        valid.append(float(line.strip().split(",")[2]))
    i+=1
fo.close()
plt.plot(train,label="d = [1,2,4]",color=color)
plt.plot(valid,color=color,ls='--')

mod_i=23
color='r'
train=[]
fo=open("training_curves/mod_"+str(mod_i)+"_train.csv",'r')
i=0
for line in fo:
    if i > 0:
        train.append(float(line.strip().split(",")[2]))
    i+=1
fo.close()
valid=[]
fo=open("training_curves/mod_"+str(mod_i)+"_valid.csv",'r')
i=0
for line in fo:
    if i > 0:
        valid.append(float(line.strip().split(",")[2]))
    i+=1
fo.close()
plt.plot(train,label="d = [1,2]",color=color)
plt.plot(valid,color=color,ls='--')

mod_i=24
color='orange'
train=[]
fo=open("training_curves/mod_"+str(mod_i)+"_train.csv",'r')
i=0
for line in fo:
    if i > 0:
        train.append(float(line.strip().split(",")[2]))
    i+=1
fo.close()
valid=[]
fo=open("training_curves/mod_"+str(mod_i)+"_valid.csv",'r')
i=0
for line in fo:
    if i > 0:
        valid.append(float(line.strip().split(",")[2]))
    i+=1
fo.close()
plt.plot(train,label="d = [1]",color=color)
plt.plot(valid,color=color,ls='--')


ax=plt.gca()
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

plt.legend()
plt.ylabel("Cross Entropy")
plt.xlabel("Epochs")
plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
plt.savefig("figures/Figure_dilation_training_curves.png",dpi=300,transparent=True)
plt.show()