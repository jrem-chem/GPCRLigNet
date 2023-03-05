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

screened=[]
fo = open("/home/jacob/More_Data/ligand_NN_2_15_21/For_Paper/pubchem_dock_results/all_scores_enriched.dat",'r')
for line in fo:
    linst=line.strip().split()
    if len(linst)>1:
        screened.append(float(linst[1]))
fo.close()

not_screened=[]
fo = open("/home/jacob/More_Data/ligand_NN_2_15_21/For_Paper/pubchem_dock_results/all_scores_unriched.dat",'r')
for line in fo:
    linst=line.strip().split()
    if len(linst)>1:
        not_screened.append(float(linst[1]))
fo.close()
screened=np.array(screened)
not_screened=np.array(not_screened)
print("screened",screened.shape)
print("not_screened",not_screened.shape)
print("screened",np.average(screened))
print("not_screened",np.average(not_screened))
print("screened",np.sum(screened < -11))
print("not_screened",np.sum(not_screened < -11))
#exit()
print("loaded")
fig,axes=plt.subplots(ncols=1,nrows=1,figsize=(3, 3))
plt.hist(screened,bins=100,alpha=0.5,color='k',label="GPCRLigNet",density=True)
plt.hist(not_screened,bins=100,alpha=0.5,color='g',label="Random",density=True)
plt.legend(framealpha=1.)
plt.xlabel("Docking Score / kcal/mol")
plt.ylabel("PDF")

ax=plt.gca()

ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.tight_layout()
plt.savefig("figures/Figure_pubchem_hist_legend.png",transparent=True,dpi=300)
#plt.show()
plt.close()

fig,axes=plt.subplots(ncols=1,nrows=1,figsize=(3, 3))
plt.hist(screened,bins=100,alpha=0.5,color='k',label="GPCRLigNet",density=True)
plt.hist(not_screened,bins=100,alpha=0.5,color='g',label="Random",density=True)
plt.legend(framealpha=1.)
plt.xlabel("Docking Score / kcal/mol")
plt.ylabel("")
plt.yticks([])
ax=plt.gca()
ax.set_xlim([-12,1])
ax.spines['left'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.tight_layout()
plt.savefig("figures/Figure_pubchem_hist.png",transparent=True,dpi=300)
#plt.show()
plt.close()
#compute fraction of data with score < x
n_do=30
score_range=np.linspace(np.min(np.concatenate([screened,not_screened],axis=0)),-8.,n_do)
fraction_screened=[100*screened[screened<=x].shape[0]/screened.shape[0] for x in score_range]
fraction_not_screened=[100*not_screened[not_screened<=x].shape[0]/not_screened.shape[0] for x in score_range]
fig,axes=plt.subplots(ncols=1,nrows=1,figsize=(2, 2))
plt.semilogy(score_range,fraction_not_screened,color='g',label="Random")
plt.semilogy(score_range,fraction_screened,color='k',label="GPCRLigNet")
#plt.legend()
plt.xlabel("Docking Score / kcal/mol")
plt.ylabel("CDF (%) ")
plt.xticks([-14,-12,-10,-8])
for x in [-12.9,-11.75,-11.,-10,-9]:
    close_x_i=np.argmin(np.abs(score_range-x))
    plt.text(score_range[close_x_i]-.5,fraction_screened[close_x_i]*5.,"{:2.1f}".format(fraction_screened[close_x_i]/fraction_not_screened[close_x_i]))
#plt.xtick_labels([-14,-12,-10,-8])
ax=plt.gca()
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.tight_layout()
plt.savefig("figures/Figure_pubchem_inset.png",transparent=True,dpi=300)
plt.close()