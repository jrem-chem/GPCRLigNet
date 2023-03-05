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
    'size'   : 10,
    'style'  : 'normal'}


mpl.rcParams['xtick.direction'] = 'in'
mpl.rcParams['ytick.direction'] = 'in'
mpl.rc('font', **font)
fpt_types=["rdkfs","maccs","cicular4","cicular6","cicular8","circularfeat4","circularfeat6","circularfeat8"]#,"random"]
#mod_inds=[10,11,12,13,14,15,16,17,18]

f_dic0={}
t_dic0={}

f_dic1={}
t_dic1={}

f_dic2={}
t_dic2={}

for i,tyype in enumerate(fpt_types):
    #if tyype=="cicular4":
        #stuff=np.load("trained_models/roc_dat_"+str(mod_inds[i])+".npy")
        stuff=np.load("/home/jacob/More_Data/ligand_NN_2_15_21/roc_compare/roc_dat_"+str(tyype)+".npy")
        false_poss=stuff[0]
        true_poss=stuff[1]
        f_dic0[tyype]=false_poss
        t_dic0[tyype]=true_poss

        stuff=np.load("/home/jacob/More_Data/ligand_NN_2_15_21/cicular_4_models_6_3_22/roc_dat_"+str(tyype)+"_1.npy")
        false_poss=stuff[0]
        true_poss=stuff[1]
        f_dic1[tyype]=false_poss
        t_dic1[tyype]=true_poss
        #
        stuff=np.load("/home/jacob/More_Data/ligand_NN_2_15_21/cicular_4_models_6_3_22/roc_dat_"+str(tyype)+"_2.npy")
        false_poss=stuff[0]
        true_poss=stuff[1]
        f_dic2[tyype]=false_poss
        t_dic2[tyype]=true_poss
        

colors=['b','orange','green','red','purple','brown','pink','grey']

fig,axes=plt.subplots(ncols=2,nrows=1,figsize=(6, 3))
axes[0].plot([0,1],[0,1],color='k')
for i,tyype in enumerate(fpt_types):
        
    if tyype[:4]=="cicu":
        lab_type="circular"+tyype[-1]
    else:
        lab_type=tyype
    axes[0].plot(np.average([f_dic0[tyype],f_dic1[tyype],f_dic2[tyype]],axis=0),np.average([t_dic0[tyype],t_dic1[tyype],t_dic2[tyype]],axis=0),label=lab_type,color=colors[i])#,marker='o'
    #axes[0].errorbar(np.average([f_dic0[tyype],f_dic1[tyype],f_dic2[tyype]],axis=0),np.average([t_dic0[tyype],t_dic1[tyype],t_dic2[tyype]],axis=0),xerr=np.std([f_dic0[tyype],f_dic1[tyype],f_dic2[tyype]],axis=0),yerr=np.std([t_dic0[tyype],t_dic1[tyype],t_dic2[tyype]],axis=0),color=colors[i])
##stuff=np.load("trained_models/new_roc_dat_5_1.npy")
#stuff=np.load("roc_compare/roc_dat_"+str("graph")+".npy")
#false_poss=stuff[0]
#true_poss=stuff[1]
#plt.plot(false_poss,true_poss,marker='o',label="graph")

#stuff=np.load("trained_models/new_roc_dat_"+str(5)+"_"+str(1)+".npy")
#false_poss=stuff[0]
#true_poss=stuff[1]
#plt.plot(false_poss,true_poss,marker='o',label="GCN")

axes[0].set_xlabel("FPR")
axes[0].set_ylabel("TPR")
axes[0].legend()

loc_0=1.
bw=.25
angle=45
for i,tyype in enumerate(fpt_types):
    auc0=get_auc_trap(f_dic0[tyype],t_dic0[tyype])
    auc1=get_auc_trap(f_dic1[tyype],t_dic1[tyype])
    auc2=get_auc_trap(f_dic2[tyype],t_dic2[tyype])
    axes[1].bar(loc_0,np.average([auc0,auc1,auc2]),width=bw*.9,yerr=np.std([auc0,auc1,auc2]),color=colors[i])
    if tyype[:4]=="cicu":
        lab_type="circular"+tyype[-1]
    else:
        lab_type=tyype
    axes[1].text(loc_0-.05,.81,lab_type,rotation='vertical')
    #axes[1].text(loc_0,np.average([auc0,auc1,auc2])+.0005,str(np.average([auc0,auc1,auc2]))[0:7], rotation=angle, rotation_mode='anchor')
    loc_0+=bw
axes[1].set_xlabel("")
axes[1].set_xticks([])
axes[1].set_ylabel("AUC")
axes[1].set_ylim(0.8,1.)
axes[1].yaxis.set_label_position("right")
axes[1].yaxis.tick_right()
plt.tight_layout()
plt.savefig("figures/Fig_ROC.png",transparent=True,dpi=300)
plt.show()
