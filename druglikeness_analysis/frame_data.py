from rdkit.Chem import QED
from rdkit import Chem
from rdkit.Chem import Lipinski
from rdkit.Chem import Descriptors
from rdkit.Chem.rdchem import Mol
from rdkit.Chem import Crippen
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

############################### Lipinski Ro5 ##############################
# Binary function to see if mol passes Linpinski's rule of 5
# OUTPUT: 1 = druglike
#       : 0 = NOT druglike 
def Ro5(mol):
    strikes=0
    if Lipinski.NumHDonors(mol)>5:
        strikes+=1
    if Lipinski.NumHAcceptors(mol)>10:
        strikes=+1
    if Descriptors.ExactMolWt(mol)>499:
        strikes+=1
    if Descriptors.MolLogP(mol)>4.99:
        strikes+=1
    if strikes>1:
        return "Not Druglike"
    elif strikes<=1:
        return "Druglike"

#################################### Veber #########################################
# OUTPUT: 1 = druglike
#       : 0 = NOT druglike 
def veber(HBD,HBA,ROTB):
    if HBD+HBA<=12:
        if ROTB<=10:
            return "Druglike"
        else: return "Not Druglike"
    else: return "Not Druglike"

##################################### Ghose #########################################
# OUTPUT: 1 = druglike
#       : 0 = NOT druglike 
def ghose(MW, ALOGP, MR, ATOM):
    strike = 0
    if MW <= 160 or MW >= 480:
        strike = strike + 1
    if ALOGP <= -0.4 or ALOGP >= 5.6:
        strike = strike + 1
    if MR <= 40 or MR >= 130:
        strike = strike + 1
    if ATOM <= 20 or ATOM >= 70:
        strike = strike + 1
    if strike == 0:
        return "Druglike"
    else: 
        return "Not Druglike"
########################## Program Body ##################################
# need to load in GPCR predictions to harvest activity data
smiles=np.load("numpy_objs/smiles.npy",allow_pickle=True)
preds=np.load("numpy_objs/GPCR_predictions.npy",allow_pickle=True)

# This for loop will be generating 4 seperate lists for each chemical scored in preds: mols, QED scores, Lipinski, and the 1st activity score (not 1-activity score)
mols=[]
activity=[]
qed=[]
lip=[]
# Molar refractivity
MR=[]
# Total # of atoms
n_atom=[]
for i in np.arange(len(preds)): # len(preds) is equal to len(mols) FYI
    activity.append(preds[i][0])
    mols.append(Chem.MolFromSmiles(smiles[i]))
    qed.append(QED.default(mols[i]))
    lip.append(Ro5(mols[i]))
    MR.append(Crippen.MolMR(mols[i]))
    n_atom.append(Mol.GetNumAtoms(mols[i]))

# should return true
len(activity)==len(mols)==len(qed)==len(preds)==len(lip)


# Creating the dataframe with all our our features for analysis
df=pd.DataFrame(columns=["MW", "ALOGP", "HBA", "HBD", "PSA", "ROTB", "AROM", "ALERTS","QED"])

for i in np.arange(len(mols)):
   row=list(QED.properties(mols[i]))
   row.append(QED.default(mols[i]))
   df.loc[len(df.index)]=row

df['MR']=MR
df['ATOM']=n_atom
df['SMILES']=smiles
df['GPCR_act']=activity
df["Lipinski"]=lip

# Loop to now add Veber and Ghose
# has to be after we had MR and ATOM (b/c we use the columns of df to caculate these)
veb=[]
gho=[]
for i in range(0,len(df)):
    veb.append(veber(df.loc[i,"HBD"],df.loc[i,"HBA"],df.loc[i,"ROTB"]))
    gho.append(ghose(df.loc[i,"MW"],df.loc[i,"ALOGP"],df.loc[i,"MR"],df.loc[i,"ATOM"]))
df['Veber']=veb
df['Ghose']=gho

column_order=['SMILES','GPCR_act','MR','ATOM',"MW", "ALOGP", "HBA", "HBD", "PSA", "ROTB", "AROM", "ALERTS","QED","Lipinski","Veber","Ghose"]
df=df.reindex(columns=column_order)

# create a column that describes each row as either QED>0.5 or QED<0.5
qed_dist=[]
for i in range(0,len(df)):
    if df.loc[i,"QED"]>0.50000:
        qed_dist.append("QED > 0.5")
    elif df.loc[i,"QED"]<0.50000:
        qed_dist.append("QED < 0.5")
    else: qed_dist.append(None)
df["qed_dist"]=qed_dist

##### AT THIS POINT OUR DATAFRAME IS COMPLETE ########

df.to_pickle("numpy_objs/framed_data.zip")