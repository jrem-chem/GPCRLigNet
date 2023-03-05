import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import DataStructs
import multiprocessing as mp
from rdkit.Chem import Draw
import matplotlib.pyplot as plt

def find_act_mol_with_bit(bit_i,all_the_smiles,all_actives):
    max_nats=80
    pos_at=[1,6,7,8,9,11,15,16,17,19,35,53]
    pos_charges=[-1,0,1]
    pos_aromatic=[True,False]
    pos_bonds=["SINGLE","DOUBLE","TRIPLE","AROMATIC"]
    #find mol with bit
    found=0
    it=0
    for iit,smile in enumerate(all_the_smiles[::-1]):
        
        #mol=Chem.MolFromSmiles(smile)
        #moll=Chem.AddHs(mol)
        #mol_s=Chem.MolToSmiles(moll)
        mol2=Chem.MolFromSmiles(smile)
        moli=Chem.AddHs(mol2)
        if moli is not None:
            if moli.GetNumAtoms()<=80 and all([x.GetAtomicNum() in pos_at for x in moli.GetAtoms()]) and all([x.GetFormalCharge() in pos_charges for x in moli.GetAtoms()]):
                arr5 = np.zeros((0,), dtype=np.float32)
                DataStructs.ConvertToNumpyArray(AllChem.GetMorganFingerprintAsBitVect(moli,4, nBits=1024),arr5)
                if arr5[bit_i]!=0.:
                    if all_actives[it,0]>=0.9:
                        out_mol=moli
                        found=1
                        break
                it+=1
    if found==0:
        print("no good",bit_i)
    return moli

def find_inact_mol_with_bit(bit_i,all_the_smiles,all_actives):
    max_nats=80
    pos_at=[1,6,7,8,9,11,15,16,17,19,35,53]
    pos_charges=[-1,0,1]
    pos_aromatic=[True,False]
    pos_bonds=["SINGLE","DOUBLE","TRIPLE","AROMATIC"]
    #find mol with bit
    found=0
    it=0
    for iit,smile in enumerate(all_the_smiles[::-1]):
        
        #mol=Chem.MolFromSmiles(smile)
        #moll=Chem.AddHs(mol)
        #mol_s=Chem.MolToSmiles(moll)
        mol2=Chem.MolFromSmiles(smile)
        moli=Chem.AddHs(mol2)
        if moli is not None:
            if moli.GetNumAtoms()<=80 and all([x.GetAtomicNum() in pos_at for x in moli.GetAtoms()]) and all([x.GetFormalCharge() in pos_charges for x in moli.GetAtoms()]):
                arr5 = np.zeros((0,), dtype=np.float32)
                DataStructs.ConvertToNumpyArray(AllChem.GetMorganFingerprintAsBitVect(moli,4, nBits=1024),arr5)
                if arr5[bit_i]!=0.:
                    if all_actives[it,0]<=0.9:
                        out_mol=moli
                        found=1
                        break
                it+=1 
    if found==0:
        print("no good",bit_i)
    return moli

smiles=[]
fo=open("/home/jacob/More_Data/ligand_NN_2_15_21/FDB-17/FDB-17-fragmentset.smi",'r')
for line in fo:
    smiles.append(line.strip())
fo.close()

n_batch=1010#1010
all_fpts=[]
all_active=[]
#load activities
for batch_i in range(n_batch):
    #input_fpt=np.load("/home/jacob/More_Data/ligand_NN_2_15_21/FDB-17/for_paper/fdb_17_"+str(batch_i)+"_cicular4.npy",allow_pickle=True)
    active_pred=np.load("/home/jacob/More_Data/ligand_NN_2_15_21/FDB-17/for_paper/fdb_17_"+str(batch_i)+"_pred.npy",allow_pickle=True)
    #all_fpts.append(input_fpt)
    all_active.append(active_pred)
all_active=np.concatenate(all_active,axis=0)



#for each bit we are interested in
#Find a mol in FDB-17 which has the bit active.
#draw it


#>10x enhacned in inactive
most_inactive_bins=[589,575,805,365,25,785]
most_inactive_bins.append(796)
#
#>10x enhacned in active
#most_active_bins=[125,228,21,15,831,146,454,503,548,960,675,664,510,270,54]
most_active_bins=[0]
#most_active_bins.append(340)
for bit in most_active_bins:
    print(bit)
    the_mol=find_act_mol_with_bit(bit,smiles,all_active)
    bi={}
    fp = AllChem.GetMorganFingerprintAsBitVect(the_mol, 4, bitInfo=bi,nBits=1024)
    mfp2_svg = Draw.DrawMorganBit(the_mol, bit, bi, useSVG=True)
    fo=open("act_inact_fpt_svgs/act_bit_"+str(bit)+".svg",'w')
    fo.write(mfp2_svg)
    fo.close()
    #exit()

for bit in most_inactive_bins:
    the_mol=find_inact_mol_with_bit(bit,smiles,all_active)
    print(bit)
    bi={}
    fp = AllChem.GetMorganFingerprintAsBitVect(the_mol, 4, bitInfo=bi,nBits=1024)

    mfp2_svg = Draw.DrawMorganBit(the_mol, bit, bi, useSVG=True)
    fo=open("act_inact_fpt_svgs/inact_bit_"+str(bit)+".svg",'w')
    fo.write(mfp2_svg)
    fo.close()