######################## Plotting QED and Activity score - histogram ##############################
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

df=pd.read_pickle("numpy_objs/framed_data.zip")
activity=list(df["GPCR_act"])
qed=list(df["QED"])

#plotting params i like to use:
#the dictionary rcParams has alot of nice things in it and you can look it its keys using .keys() to see what else you can do.
plt.rcParams['font.family'] = 'DeJavu Serif'
plt.rcParams['font.serif'] = ['Calibri']

#predefine the figure object with size
fig = plt.figure(figsize=(3,3))

#see https://matplotlib.org/stable/gallery/color/named_colors.html for colors
plt.hist2d(activity,qed,bins=10,cmap=plt.get_cmap("gist_ncar"))

#grab the axes we plotted to

#set some extra tick options for making it look nice
plt.tick_params(axis='both',direction='in', labelsize=10)
plt.xlabel("GPCR Activity Score")
plt.ylabel("QED")

#makes a nice layout
plt.tight_layout()

#save it
plt.savefig("plots/activity_vs_qed.png",dpi=300,transparent=True)
plt.close()

########################## Plotting Activity score by Ro5 pass/fail - Boxplot ######################
# lip=np.array(lip) # convert to array for easy boolean indexing
# activity=np.array(activity) # dido

# # subset of activity scores for Ro5 passes
# lip_pass=activity[lip=="Druglike"]

# # subset of activity scores for Ro5 fails
# lip_fail=activity[lip=="Not Druglike"]

# # combining the data for the box plot
# lip_data=[lip_pass,lip_fail]

# # now try to make a boxplot to show these results
# plt.boxplot(lip_data)

# #grab the axes we plotted to
# axes=plt.gca()

# #set some extra tick options for making it look nice
# axes.set_xlabel("Lipinski Pass/Fail")
# axes.set_ylabel("GPCR Activity Score")
# axes.set_xticklabels(['Pass','Fail'])

# #save it
# plt.savefig("Lipinski_boxplot",dpi=300,transparent=True)

# plt.show()

############################# Other Plots to make and analysis to do ###########################
# 1. Graph distribution of QED by activity>0.5 and activity<0.5 
#   - this could be two histograms, boxplots, violin plots, whatever else looks nice
# 2. Repeat #1 but with other drug-likness metrics, such as Lipinski, Veber, Ghose
# 3. Perform the Kolmogorov–Smirnov test (probably on scipy) to see if the two distributions are statistically different

################### Seaborn plots for distributions ################
# sns.set_theme(style="whitegrid")
# sns.violinplot(x="qed_dist",y='GPCR_act', data=df)
# plt.show()
# sns.boxenplot(x="qed_dist", y="GPCR_act", data=df)
# plt.show()

# # Activity by Lipinski
# sns.violinplot(x="Lipinski",y='GPCR_act', data=df)
# plt.show()
# sns.boxenplot(x="Lipinski", y="GPCR_act", data=df)
# plt.show()

# # Activity by Veber
# sns.violinplot(x="Veber",y='GPCR_act', data=df)
# plt.show()
# sns.boxenplot(x="Veber", y="GPCR_act", data=df)
# plt.show()

# # Activity by Ghose
# sns.violinplot(x="Ghose",y='GPCR_act', data=df)
# plt.show()
# sns.boxenplot(x="Ghose", y="GPCR_act", data=df)
# plt.show()

################## 3. Performing KS test ##################
### have to split data frame first
splitQ = df.groupby("qed_dist")
splitLip = df.groupby("Lipinski")
splitVeb = df.groupby("Veber")
splitGho = df.groupby("Ghose")

### split by distribution half
# QED
upperQ = splitQ.get_group("QED > 0.5")
lowerQ = splitQ.get_group("QED < 0.5")

# Lipinski
passLip = splitLip.get_group("Druglike")
failLip = splitLip.get_group("Not Druglike")

# Veber
passVeb = splitVeb.get_group("Druglike")
failVeb = splitVeb.get_group("Not Druglike")

# Ghose
passGho = splitGho.get_group("Druglike")
failGho = splitGho.get_group("Not Druglike")

### grab GPCR activity for each half as array (for ks_2samp)
# QED
upQ_act=np.array(upperQ['GPCR_act'])
lowQ_act=np.array(lowerQ['GPCR_act'])

# Lipinski
passLip_act=np.array(passLip['GPCR_act'])
failLip_act=np.array(failLip['GPCR_act'])

# Veber
passVeb_act=np.array(passVeb['GPCR_act'])
failVeb_act=np.array(failVeb['GPCR_act'])

# Ghose
passGho_act=np.array(passGho['GPCR_act'])
failGho_act=np.array(failGho['GPCR_act'])

#### perform the test
# QED
q_stat,q_pval=stats.ks_2samp(upQ_act,lowQ_act)

# Lipinski
lip_stat,lip_pval=stats.ks_2samp(passLip_act,failLip_act)

# Veber
veb_stat,veb_pval=stats.ks_2samp(passVeb_act,failVeb_act)

# Ghose
gho_stat,gho_pval=stats.ks_2samp(passGho_act,failGho_act)


########### Overlapping Histograms ############
defaults=['tab:blue','tab:orange']

# QED
plt.figure(figsize=(3,3))
plt.hist(upQ_act, bins=100, alpha=0.5, label="QED > 0.5",color=defaults[0])
plt.hist(lowQ_act, bins=100, alpha=0.5, label="QED < 0.5",color=defaults[1])
plt.xlabel("Activity Score", size=14)
plt.ylabel("Count", size=14)
plt.title("QED, p="+f"{q_pval:.10g}")
plt.legend(loc='upper right')
plt.axvline(upQ_act.mean(),linestyle='--',color=defaults[0])
plt.axvline(lowQ_act.mean(),linestyle='--',color=defaults[1])
plt.tight_layout()
plt.savefig("plots/QED_dists.png",dpi=300,transparent=True)
plt.close()

# Lipinksi
plt.figure(figsize=(3,3))
plt.hist(passLip_act, bins=100, alpha=0.5, label="Druglike",color=defaults[0])
plt.hist(failLip_act, bins=100, alpha=0.5, label="Not Druglike",color=defaults[1])
plt.xlabel("Activity Score", size=14)
plt.ylabel("Count", size=14)
plt.title("Lipinski Ro5, p="+f"{lip_pval:.10g}")
plt.legend(loc='upper right')
plt.axvline(passLip_act.mean(),linestyle='--',color=defaults[0])
plt.axvline(failLip_act.mean(),linestyle='--',color=defaults[1])
plt.tight_layout()
plt.savefig("plots/Lip_dists.png",dpi=300,transparent=True)
plt.close()

# Veber
plt.figure(figsize=(3,3))
plt.hist(passVeb_act, bins=100, alpha=0.5, label="Druglike",color=defaults[0])
plt.hist(failVeb_act, bins=100, alpha=0.5, label="Not Druglike",color=defaults[1])
plt.xlabel("Activity Score", size=14)
plt.ylabel("Count", size=14)
plt.title("Veber method, p="+f"{veb_pval:.10g}")
plt.legend(loc='upper right')
plt.axvline(passVeb_act.mean(),linestyle='--',color=defaults[0])
plt.axvline(failVeb_act.mean(),linestyle='--',color=defaults[1])
plt.tight_layout()
plt.savefig("plots/Veb_dists.png",dpi=300,transparent=True)
plt.close()

# Ghose
plt.figure(figsize=(3,3))
plt.hist(passGho_act, bins=100, alpha=0.5, label="Druglike",color=defaults[0])
plt.hist(failGho_act, bins=100, alpha=0.5, label="Not Druglike",color=defaults[1])
plt.xlabel("Activity Score", size=14)
plt.ylabel("Count", size=14)
plt.title("Ghose method, p="+f"{gho_pval:.10g}")
plt.legend(loc='upper right')
plt.axvline(passGho_act.mean(),linestyle='--',color=defaults[0])
plt.axvline(failGho_act.mean(),linestyle='--',color=defaults[1])
plt.tight_layout()
plt.savefig("plots/Gho_dists.png",dpi=300,transparent=True)
plt.close()

print("QED Pval=",q_pval,"LIP Pval=",lip_pval,"VEB Pval=",veb_pval,"GHO Pval=",gho_pval)

# The p-values are so infintesimally small (due in part to the massive sample size) they just show up as 0
# https://stackoverflow.com/questions/20530138/scipy-p-value-returns-0-0