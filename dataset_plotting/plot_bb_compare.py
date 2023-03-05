import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl


font = {'family' : 'Arial',
    #'weight' : 'medium',
    'size'   : 9,
    'style'  : 'normal'}


mpl.rcParams['xtick.direction'] = 'in'
mpl.rcParams['ytick.direction'] = 'in'
mpl.rc('font', **font)

bb_act=np.load("/home/jacob/More_Data/ligand_NN_2_15_21/bb_compounds/bb_cicular4_nkis.npy")
bb_activities=np.load("/home/jacob/More_Data/ligand_NN_2_15_21/bb_compounds/bb_circular4_pred_act.npy")
bb_act_fixed=[]
print(bb_act)


for i in range(bb_activities.shape[0]):
    print(bb_act[i])
    if "%" in bb_act[i]:
        bb_act_fixed.append(1000.)
    elif bb_act[i] =="NA":
        bb_act_fixed.append(1000.)
    else:
        bb_act_fixed.append(float(bb_act[i]))

# plt.figure(figsize=[3,3])
# plt.scatter(bb_act_fixed,bb_activities[:,0],color='k')

# #bb_inds=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,29,30,31,32,33,34,35,36]
# #for i in range(bb_act.shape[0]):
# #    plt.text(bb_act_fixed[i]-10,bb_activities[i,0]+.01,bb_inds[i])

# plt.xlabel("Ki / nM")
# plt.ylabel("Activity Prediciton")
# ax=plt.gca()
# ax.spines['right'].set_visible(False)
# ax.spines['top'].set_visible(False)
# plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
# #plt.savefig("figures/Figure_beebe_PAC1.png",dpi=300,transparent=True)
# plt.show()


#compute confusion matrix
FP=0
TP=0
FN=0
TN=0
bb_act_fixed=np.array(bb_act_fixed)
for i in range(bb_act_fixed.shape[0]):
    #FP
    if bb_act_fixed[i]>= 1000. and bb_activities[i,0]>0.5:
        FP+=1
    #TN
    elif bb_act_fixed[i]>= 1000. and bb_activities[i,0]<=0.5:
        TN+=1
    #TP
    elif bb_act_fixed[i]< 1000. and bb_activities[i,0]>0.5:
        TP+=1
    #FN
    elif bb_act_fixed[i]< 1000. and bb_activities[i,0]<=0.5:
        FN+=1


plt.figure(figsize=[3,3])

print('FP',FP)
print('TP',TP)
print('FN',FN)
print('TN',TN)

confus=np.array([[TN,FN],[FP,TP]])
plt.imshow(confus,cmap='Greens')
plt.colorbar()
plt.xlabel("Experiment Active")
plt.ylabel("Pred. Active")
ax=plt.gca()
ax.set_xticks([0,1])
ax.set_xticklabels(['False','True'])
ax.set_yticks([0,1])
ax.set_yticklabels(['False','True'])
for i in [0,1]:
    for j in [0,1]:
        if i==1:
            plt.annotate(str(int(confus[i,j])),[j,i],color='white')
        else:
            plt.annotate(str(int(confus[i,j])),[j,i],color='black')
plt.tight_layout()
plt.savefig("figures/Figure_beebe_PAC1_confus.png",dpi=300,transparent=True)
plt.show()