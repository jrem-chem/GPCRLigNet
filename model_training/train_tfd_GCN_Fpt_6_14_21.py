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
def load_the_data_sparse(tmod):
    def load_array_type(data_type,tmod,shape):
        print("loading",data_type)
        root="data_sets/new_data_6_13_21/GLASS/"
        n_per=10000
        n_dat=63
        dats=[]
        for i in range(0,n_dat*n_per,n_per):
            #print(i)
            indices=np.load(root+data_type+"_i_"+str(i)+".npy",allow_pickle=True)[::tmod]
            values=np.load(root+data_type+"_v_"+str(i)+".npy",allow_pickle=True)[::tmod]
            #shape=[max_atoms,dim_feat]
            for j in range(len(indices)):
                sparse_dat=tf.SparseTensor(indices=indices[j],
                  values=values[j].astype(np.float32),
                  dense_shape=shape)
                sparse_dat=tf.sparse.reorder(sparse_dat)
                dats.append(tf.sparse.expand_dims(sparse_dat, axis=0))
        root="data_sets/new_data_6_13_21/DUDE/"
        n_per=10000
        n_dat=8
        for i in range(0,n_dat*n_per,n_per):
            #print(i)
            indices=np.load(root+data_type+"_i_"+str(i)+".npy",allow_pickle=True)[::tmod]
            values=np.load(root+data_type+"_v_"+str(i)+".npy",allow_pickle=True)[::tmod]
            #shape=[max_atoms,dim_feat]
            for j in range(len(indices)):
                sparse_dat=tf.SparseTensor(indices=indices[j],
                  values=values[j].astype(np.float32),
                  dense_shape=shape)
                sparse_dat=tf.sparse.reorder(sparse_dat)
                dats.append(tf.sparse.expand_dims(sparse_dat, axis=0))
        dats=tf.sparse.concat(0,dats)
        return dats
    #nod_mat
    #lin_mat
    #mol_act
    #edg_mat
    #adj_mat
    #load the nod_mats
    data_type="mol_act"
    nod_mat=load_array_type("nod_mat",tmod,[max_atoms,dim_feat])
    #lin_mat=load_array_type("lin_mat",tmod,[max_n_bonds,max_n_bonds])
    #edg_mat=load_array_type("edg_mat",tmod,[max_n_bonds,dim_edge_feat])
    adj_mat=load_array_type("adj_mat",tmod,[max_atoms,max_atoms])
    #mol_act=load_array_type("mol_act",tmod,[])
    #n_per=10000
    #n_dat=63
    #root="data_sets/new_data_6_13_21/GLASS/"
    #mol_act=[]
    #for i in range(0,n_dat*n_per,n_per):
    #    acts=np.load(root+"mol_act_"+str(i)+".npy")[::tmod]
    #    #[f(x) if condition else g(x) for x in sequence]
    #    mol_act.extend([[1,0] if x < 1e3 else [0,1] for x in acts])
    #
    #n_per=10000
    #n_dat=8
    #root="data_sets/new_data_6_13_21/DUDE/"
    #for i in range(0,n_dat*n_per,n_per):
    #    acts=np.load(root+"mol_act_"+str(i)+".npy")[::tmod]
    #    #[f(x) if condition else g(x) for x in sequence]
    #    mol_act.extend([[1,0] if x < 1e3 else [0,1] for x in acts])
    #mol_act=np.array(mol_act).astype(np.float32)
    return nod_mat,adj_mat #=load_the_data_sparse(tmod)
    #return nod_mat,lin_mat,edg_mat,adj_mat,mol_act

def gen_roc(predicted,truth,cutts):
    false_poss=[]
    false_negss=[]
    true_poss=[]
    true_negss=[]
    for cutt in cutts:
        false_pos=0.
        false_neg=0.
        true_pos=0.
        true_neg=0.
        P=0.
        N=0.
        temp_pred = np.zeros_like( predicted ) + predicted
        temp_pred [ temp_pred >  cutt ] = 1.
        temp_pred [ temp_pred <= cutt ] = 0.
        #print(cutt,temp_pred[0:5])
        for i,pred in enumerate(temp_pred):
            if pred[0]==1. and  truth[i,0] ==0.:
                false_pos+=1.
            elif pred[0]==0. and truth[i,0]==1.:
                false_neg+=1.
            elif pred[0]==0. and truth[i,0]==0.:
                true_neg+=1.
            elif pred[0]==1. and truth[i,0]==1.:
                true_pos+=1.
            if truth[i,0]==1.:
                P+=1.
            else:
                N+=1.
        #print(false_pos+false_neg+true_neg+true_pos,N,P)
        false_poss.append(false_pos/(false_pos+true_neg))#N)
        false_negss.append(false_neg/P)
        true_poss.append(true_pos/(true_pos+false_neg))#P)
        true_negss.append(true_neg/N)
    #fig,axes=plt.subplots(ncols=2)
    plt.plot(false_poss,true_poss,marker='o')
    #axes[1].plot(false_negss,true_negss,marker='o')
    plt.plot([0,1],[0,1],color='k')
    for i in range(len(cutts)):
        plt.annotate(str(cutts[i])[0:4],[false_poss[i]-.1,true_poss[i]])
    #axes[1].plot([0,1],[0,1],color='k')
    plt.xlabel("false positive rate")
    plt.ylabel("true positive rate")
    #axes[1].set_xlabel("false negative rate")
    #axes[1].set_ylabel("true negative rate")
    plt.show()
    np.save("trained_models/roc_"+str(model_i)+"_dat",np.array([false_poss,true_poss]))

def load_numpy_tot(tmod,name):
    indices=np.load("data_sets/"+str(name)+"_i_"+str(tmod)+".npy")
    values=np.load("data_sets/"+str(name)+"_v_"+str(tmod)+".npy")
    shape=np.load("data_sets/"+str(name)+"_s_"+str(tmod)+".npy")
    sparse_dat=tf.SparseTensor(indices=indices,
                  values=values,
                  dense_shape=shape)
    return sparse_dat
def load_the_data(tmod,typee):
    def load_array_type(data_type,tmod):
        root="data_sets/new_data_6_13_21/GLASS/"
        n_per=10000
        n_dat=63
        dats=[]
        for i in range(0,n_dat*n_per,n_per):
            x=np.load(root+"basic_fpt_"+data_type+"_"+str(i)+".npy")
            dats.append(x.astype(np.float32))
        root="data_sets/new_data_6_13_21/DUDE/"
        n_per=10000
        n_dat=8
        for i in range(0,n_dat*n_per,n_per):
            x=np.load(root+"basic_fpt_"+data_type+"_"+str(i)+".npy")
            dats.append(x.astype(np.float32))
        dats=np.concatenate(dats,axis=0)
        return dats
    #nod_mat
    #lin_mat
    #mol_act
    #edg_mat
    #adj_mat
    #load the nod_mats
    fpts_dats=load_array_type(typee,tmod)
    
    #mol_act=load_array_type("mol_act",tmod,[])
    n_per=10000
    n_dat=63
    root="data_sets/new_data_6_13_21/GLASS/"
    mol_act=[]
    for i in range(0,n_dat*n_per,n_per):
        acts=np.load(root+"mol_act_"+str(i)+".npy")[::tmod]
        #[f(x) if condition else g(x) for x in sequence]
        mol_act.extend([[1,0] if x < 1e3 else [0,1] for x in acts])
    
    n_per=10000
    n_dat=8
    root="data_sets/new_data_6_13_21/DUDE/"
    for i in range(0,n_dat*n_per,n_per):
        acts=np.load(root+"mol_act_"+str(i)+".npy")[::tmod]
        #[f(x) if condition else g(x) for x in sequence]
        mol_act.extend([[1,0] if x < 1e3 else [0,1] for x in acts])
    mol_act=np.array(mol_act).astype(np.float32)

    return fpts_dats,mol_act
class graph_layer(tf.keras.layers.Layer):
    #this layer applys a the initial convolution to each node in a graph to output features for each node
    def __init__(self,num_inputs,num_outputs,reg_rate):
        super(graph_layer, self).__init__()
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.reg_rate=reg_rate
    def build(self,input_shape):
        self.param = self.add_weight("param",
                                    shape=[self.num_inputs,self.num_outputs], regularizer=tf.keras.regularizers.l2(l2=self.reg_rate))
        self.bias = self.add_weight("bias",
                                    shape=[self.num_outputs])
    def call(self, adj_matrix,node_feats):
        #compute edge node_feats *node_feats.transpose()
        #adj_matrix=inputz[0]#assumes both are not sparse
        #node_feats=inputz[1]#assumes both are not sparse
        #apply convolution
        conv_feats=tf.linalg.matmul(adj_matrix,node_feats)
        #compute outputs
        return tf.keras.activations.elu(tf.linalg.matmul(conv_feats,self.param)+self.bias)

class graph_layer2(tf.keras.layers.Layer):
    #this layer applys a the initial convolution to each node in a graph to output features for each node
    #computes activation(AF(i)W1 + F(i-1)W2)
    def __init__(self,num_inputs,num_outputs,num_inputsi_m1,reg_rate):
        super(graph_layer2, self).__init__()
        self.num_inputsi_m1 = num_inputsi_m1
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.reg_rate=reg_rate
    def build(self,input_shape):
        self.param1 = self.add_weight("param1",
                                    shape=[self.num_inputs,self.num_outputs], regularizer=tf.keras.regularizers.l2(l2=self.reg_rate))
        self.param2 = self.add_weight("param2",
                                    shape=[self.num_inputsi_m1,self.num_outputs], regularizer=tf.keras.regularizers.l2(l2=self.reg_rate))
        self.bias = self.add_weight("bias",
                                    shape=[self.num_outputs])
    def call(self, adj_matrix,node_featsi,node_featsi_m1):
        #compute edge node_feats *node_feats.transpose()
        #adj_matrix=inputz[0]#assumes both are not sparse
        #node_feats=inputz[1]#assumes both are not sparse
        #apply convolution
        conv_feats=tf.linalg.matmul(adj_matrix,node_featsi)
        #compute outputs
        return tf.keras.activations.elu(tf.linalg.matmul(conv_feats,self.param1)+tf.linalg.matmul(node_featsi_m1,self.param2)+self.bias)

class graph_dilations(tf.keras.layers.Layer):
    #this layer generates n_output dilations of an adjacency matrix
    def __init__(self,output_dilations):
        super(graph_dilations, self).__init__()
        self.output_dilations = output_dilations

    def build(self,input_shape):
        self.param = None#

    def call(self, adj_matrix):
        #print("adj_matrix",adj_matrix.shape)
        #A=tf.sparse.to_dense(adj_matrix)
        #print("A",A.shape)
        #print(0,A)
        eeye=tf.eye(tf.shape(adj_matrix)[1],batch_shape=[tf.shape(adj_matrix)[0]])
        #print("eeye",eeye.shape)
        xs=[adj_matrix]
        #print(adj_matrix)
        for k in self.output_dilations:
            #if k ==0:
            #    x=tf.clip_by_value(tf.clip_by_value(tf.linalg.matmul(xs[-1]-eeye,xs[0]-eeye),0,1)+eeye,0,1)
            #else:
            x=tf.clip_by_value(tf.clip_by_value(tf.clip_by_value(tf.linalg.matmul(xs[-1]-eeye,xs[0]-eeye),0,1)-tf.reduce_sum(tf.stack(xs,axis=-1),axis=-1),0,1)+eeye,0,1)
            xs.append(x)
            #print(x)
        ##print("degres")
        #normalize the adjacency based on degree.
        #https://arxiv.org/abs/1609.02907
        normxs=[]
        for x in xs:
           degree_mat=tf.math.sqrt(tf.math.reciprocal_no_nan(tf.linalg.diag(tf.reduce_sum(tf.clip_by_value(x-eeye,0,1000000),axis=1))))
           #print(degree_mat)
           #normxs.append(tf.clip_by_value(tf.linalg.matmul(degree_mat,tf.linalg.matmul(x,degree_mat))+tf.clip_by_value(tf.linalg.diag(tf.linalg.diag_part(x)+1),0,1),0,1))
           normxs.append(tf.linalg.matmul(degree_mat,tf.linalg.matmul(x,degree_mat)))
           #print(normxs[-1])
        return normxs

def crossentropy(y_true,y_pred):
    return tf.reduce_mean(-1.*y_true[:,0]*tf.math.log(tf.clip_by_value(y_pred[:,0],1e-10,1.)) - (y_true[:,1])*tf.math.log(tf.clip_by_value(y_pred[:,1],1e-10,1.)))

def sigmoid_cross_entropy_with_logits_loss(y_true,y_pred):
    return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=y_true,logits=y_pred))

def tanimoto_loss(y_true,y_pred):
    M11=tf.reduce_sum(tf.multiply(y_true, y_pred),axis=-1)#tf.dot(y_true*y_pred)
    M01=tf.reduce_sum(tf.clip_by_value(y_true + y_pred,0.,1.),axis=-1)#tf.dot(y_true*y_true)
    #M10=tf.reduce_sum(tf.multiply(y_true, 1-y_pred),axis=-1)
    #M00=tf.reduce_sum(tf.multiply(1-y_true, 1-y_pred),axis=-1)
    return tf.reduce_mean((M01-M11)/M01)


def tanimoto_score(vec1, vec2):
    '''
    Return the Tanimoto score between vec1 and vec2. 
    Args:
        vec1, vec2: The two vectors to find the Tanimoto coefficient for. MUST be
                    of the same length
    
    No validation is performed except same length checks. It is assumed that
    the caller passes properly weighted data.

    '''
    N = len(vec1)
    
    assert N == len(vec2)
    
    v1v2, v1v1, v2v2 = 0., 0., 0.
    for i in range(N):
        v1v2 += vec1[i] * vec2[i]
        v1v1 += vec1[i] * vec1[i]
        v2v2 += vec2[i] * vec2[i]

    return v1v2 / (v1v1 + v2v2 - v1v2)


#ytrue=np.array([[0,1,1],[0,0,1]]).astype(np.float32)
#ypred=np.array([[0,1,1],[0,.1,1]]).astype(np.float32)
#print(tanimoto_loss(ytrue,ypred))
##print(tanimoto_score(ytrue[1],ypred[1]))
#exit()


model_config={}
max_atoms=80
dim_feat=17#17
max_n_bonds=112
dim_edge_feat=21

fpt_type=sys.argv[1]#"cicular6"

fpt_sizes={}
fpt_sizes["rdkfs"]=2048
fpt_sizes["maccs"]=167
fpt_sizes["cicular4"]=1024
fpt_sizes["cicular6"]=1024
fpt_sizes["cicular8"]=1024
fpt_sizes["circularfeat4"]=1024
fpt_sizes["circularfeat6"]=1024
fpt_sizes["circularfeat8"]=1024

glob_rseed=18746
tf.random.set_seed(glob_rseed)
model_config["glob_rseed"]=glob_rseed

loca_rseed=512
model_config["loca_rseed"]=loca_rseed

n_g_conv=4
reg_rates=[0.0,0,0,0]#[.005,.001,.00,.00,.000,00,00,00]#[.01,.0075,.005,.0025,.000]#[.01,.008,.006,.004,.002,.000,0.,0.,0.]
reg_rates=[x*1. for x in reg_rates]
model_config["n_g_conv"]=n_g_conv

#g_conv_filts=[int(x) for x in np.linspace(300,100,n_g_conv)]#[int(x) for x in np.linspace(300,200,n_g_conv)]#[300,275,250,225,200,200,200,175]#[300,250,200,150]
g_conv_filts=[300 for x in range(n_g_conv)]
model_config["g_conv_filts"]=g_conv_filts


dense_widths=[750,250,2]#[int(x) for x in np.linspace(200*(g_conv_filts[-1]),2,n_dense+1)]

model_config["dense_widths"]=dense_widths

dilations=[1,3,5]#[]#
n_dilations=len(dilations)
model_config["dilations"]=dilations

epochs= 500
model_config["epochs"]=epochs

batch=100#1000
model_config["batch"]=batch

learn_r=2e-4#1e-1#5e-5
model_config["learn_r"]=learn_r

d_rate=.25#1
model_config["d_rate"]=d_rate

retrain=1

tmod=int(sys.argv[2])
model_config["tmod"]=tmod

model_i=6
model_config["model_i"]=model_i

v_split=0.3
model_config["v_split"]=v_split


print("Loading Data")
fpts_dats,all_mol_act=load_the_data(1,fpt_type)
#plt.plot(fpts_dats[0])
#plt.show()
#print(keras.losses.MeanSquaredError()(fpts_dats,0.*np.ones_like(fpts_dats)))#0.*np.ones_like(fpts_dats)))
#all_nod_mat,all_adj_mat =load_the_data_sparse(tmod)
#exit()

n_max_at=80
n_at_type=17

#all_nod_mat=load_numpy_tot(tmod,"nod_mat")
###all_lin_mat=load_numpy_tot(tmod,"lin_mat")
###all_edg_mat=load_numpy_tot(tmod,"edg_mat")
#
#all_adj_mat=load_numpy_tot(tmod,"adj_mat")
#all_mol_act=load_numpy_tot(tmod,"act_mat")
#
#all_nod_mat=tf.sparse.to_dense(all_nod_mat)
#print("done")
#all_lin_mat=tf.sparse.to_dense(all_lin_mat)
#print("done")
#all_edg_mat=tf.sparse.to_dense(all_edg_mat)
#print("done")
#all_adj_mat=tf.sparse.to_dense(all_adj_mat)
#print("done")
#all_mol_act=tf.sparse.to_dense(all_mol_act)
#print("done")
#
n_dat=all_mol_act.shape[0]

#print(all_adj_mat.shape)
#print(all_nod_mat.shape)
#print(fpts_dats.shape)
##
#full_dataset=tf.data.Dataset.from_tensor_slices(((all_adj_mat,all_nod_mat),fpts_dats)).shuffle(buffer_size=n_dat,seed=loca_rseed)
#train_dataset=full_dataset.take(int((1-v_split)*n_dat))
#test_dataset=full_dataset.skip(int((1-v_split)*n_dat))
#for ele in train_dataset:
#    print(ele)
#    break
#for ele in test_dataset:
#    print(ele)
#    break
#
#
##train_dataset = tf.data.Dataset.from_tensor_slices(((all_adj_mat[:int((1-v_split)*n_dat)],all_nod_mat[:int((1-v_split)*n_dat)]),all_mol_act[:int((1-v_split)*n_dat)])).shuffle(buffer_size=n_dat,seed=loca_rseed))
##test_dataset = tf.data.Dataset.from_tensor_slices(((all_adj_mat[int((1-v_split)*n_dat):],all_nod_mat[int((1-v_split)*n_dat):]),all_mol_act[int((1-v_split)*n_dat):])).shuffle(buffer_size=n_dat,seed=loca_rseed))
#tf.data.experimental.save(train_dataset,"data_sets/train_dataset_fptmap_"+str(fpt_type)+"_"+str(tmod))
#tf.data.experimental.save(test_dataset,"data_sets/test_dataset_fptmap_"+str(fpt_type)+"_"+str(tmod))
#exit()
train_dataset=tf.data.experimental.load("data_sets/train_dataset_fptmap_"+str(fpt_type)+"_"+str(tmod),element_spec =((tf.TensorSpec(shape=( n_max_at,n_max_at), dtype=tf.float32, name=None), tf.TensorSpec(shape=( n_max_at,dim_feat), dtype=tf.float32, name=None)),tf.TensorSpec(shape=( fpt_sizes[fpt_type]), dtype=tf.float32, name=None)))
test_dataset=tf.data.experimental.load("data_sets/test_dataset_fptmap_"+str(fpt_type)+"_"+str(tmod),element_spec =((tf.TensorSpec(shape=( n_max_at,n_max_at), dtype=tf.float32, name=None), tf.TensorSpec(shape=( n_max_at,dim_feat), dtype=tf.float32, name=None)),tf.TensorSpec(shape=( fpt_sizes[fpt_type]), dtype=tf.float32, name=None)))

for adata in train_dataset:
    temp_adj=np.array([adata[0][0]])
    temp_feat=np.array([adata[0][1]])
    break

#print(temp_adj)
#print(temp_feat.shape)

#exit()
#valid_output=[]
#for data in test_dataset:
#    valid_output.append(data[-1])
#valid_output=tf.concat(valid_output,axis=0)

train_dataset=train_dataset.batch(batch)
test_dataset=test_dataset.batch(batch)



#exit()
log_dir = "log_dir/new_t_board_gcn_fptmap_"+str(fpt_type)+"_"+str(model_i)
if os.path.isdir(log_dir):
    shutil.rmtree(log_dir)
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

if retrain==1:
    #mirrored_strategy = tf.distribute.MirroredStrategy()
    #with mirrored_strategy.scope():

    input_atom_adjs = keras.Input(shape=(max_atoms,max_atoms),sparse=False)
    input_atom_feats = keras.Input(shape=(max_atoms,dim_feat),sparse=False)

    #generate dilations of atom and bond adjs
    gd=graph_dilations(dilations)
    atom_dilated_adjs=gd(input_atom_adjs)
    #bond_dilated_adjs=gd(input_bond_adjs)

    temp_adj_dial=gd(temp_adj)


    #apply GCN layers to dilated atom and bond adjs
    inp_atom_gcn_outs=[]
    #inp_bond_gcn_outs=[]
    temp_inp_atom_gcn_outs=[]

    for di in range(n_dilations+1):
        glay1=graph_layer(dim_feat,g_conv_filts[0],reg_rates[0])
        #glay2=graph_layer(dim_edge_feat,g_conv_filts[0],reg_rates[0])
        inp_atom_gcn_outs.append(glay1(atom_dilated_adjs[di],input_atom_feats))
        #inp_bond_gcn_outs.append(glay2(bond_dilated_adjs[di],input_bond_feats))
        temp_inp_atom_gcn_outs.append(glay1(temp_adj_dial[di],temp_feat))
        #plt.imshow(temp_adj_dial[di][0])
        #plt.show()
        #print(temp_inp_atom_gcn_outs[-1])
        #exit()

    atom_gcn_outs=[]
    #bond_gcn_outs=[]
    temp_gcn_outs=[]
    for gcn_i in range(n_g_conv-1):
        print("WEEEE")
        #apply GCN layers to dilated atom and bond adjs
        tmp_atom_gcn_outs=[]
        #tmp_bond_gcn_outs=[]
        temp_tmp_atom_gcn_outs=[]
        for di in range(n_dilations+1):
            if gcn_i==0:#pass first gcn to them
                tmp_atom_gcn_outs.append(graph_layer2(g_conv_filts[0],g_conv_filts[0],dim_feat,reg_rates[gcn_i+1])(atom_dilated_adjs[di],inp_atom_gcn_outs[di],input_atom_feats))
                #tmp_bond_gcn_outs.append(graph_layer(g_conv_filts[0],g_conv_filts[0],reg_rates[gcn_i+1])(bond_dilated_adjs[di],inp_bond_gcn_outs[di]))
                temp_tmp_atom_gcn_outs.append(graph_layer2(g_conv_filts[0],g_conv_filts[0],dim_feat,reg_rates[gcn_i+1])(temp_adj_dial[di],temp_inp_atom_gcn_outs[di],temp_feat))
            elif gcn_i==1:#pass last gcn to them
                tmp_atom_gcn_outs.append(graph_layer2(g_conv_filts[0],g_conv_filts[0],g_conv_filts[0],reg_rates[gcn_i+1])(atom_dilated_adjs[di],atom_gcn_outs[gcn_i-1][di],inp_atom_gcn_outs[di]))
                temp_tmp_atom_gcn_outs.append(graph_layer2(g_conv_filts[0],g_conv_filts[0],g_conv_filts[0],reg_rates[gcn_i+1])(temp_adj_dial[di],temp_gcn_outs[gcn_i-1][di],temp_inp_atom_gcn_outs[di]))
            else:
                tmp_atom_gcn_outs.append(graph_layer2(g_conv_filts[0],g_conv_filts[0],g_conv_filts[0],reg_rates[gcn_i+1])(atom_dilated_adjs[di],atom_gcn_outs[gcn_i-1][di],atom_gcn_outs[gcn_i-2][di]))
                #tmp_bond_gcn_outs.append(graph_layer(g_conv_filts[0],g_conv_filts[0],reg_rates[gcn_i+1])(bond_dilated_adjs[di],bond_gcn_outs[gcn_i-1][di]))
                temp_tmp_atom_gcn_outs.append(graph_layer2(g_conv_filts[0],g_conv_filts[0],g_conv_filts[0],reg_rates[gcn_i+1])(temp_adj_dial[di],temp_gcn_outs[gcn_i-1][di],temp_gcn_outs[gcn_i-2][di]))
        #save for next gcn_i
        atom_gcn_outs.append(tmp_atom_gcn_outs)
        #bond_gcn_outs.append(tmp_bond_gcn_outs)
        temp_gcn_outs.append(temp_tmp_atom_gcn_outs)
        

    tempor=tf.concat(atom_gcn_outs[-1],axis=-1)
    temp_tempor=tf.concat(temp_gcn_outs[-1],axis=-1)

    #print(tempor.shape)
    #atom_gcn_out=tf.squeeze(layers.MaxPooling1D(pool_size=tempor.shape[1],data_format='channels_last')(tempor),axis=1)
    atom_gcn_out=tf.reduce_mean(tempor,axis=1)
    temp_atom_gcn_out=tf.reduce_mean(temp_tempor,axis=1)
    #plt.plot(temp_atom_gcn_out[0])
    #plt.show()
    #print(atom_gcn_out.shape)
    #now predict activity from the final fingerprint
    droopout=layers.Dropout(d_rate)
    pred_temp_outd=droopout(atom_gcn_out)
    den_lay=layers.Dense(fpt_sizes[fpt_type],activation='sigmoid')#,kernel_regularizer=tf.keras.regularizers.l2(l2=.005))
    #pred_temp_out=den_lay(all_temp_fpts[-1])
    #print(tf.stack(all_temp_fpts,axis=-1).shape)
    #pred_temp_out=den_lay(tf.reduce_mean(tf.stack(all_temp_fpts,axis=-1),axis=-1))
    pred_temp_out=den_lay(pred_temp_outd)
    temp_pred_temp_out=den_lay(temp_atom_gcn_out)
    #plt.plot(temp_pred_temp_out[0])
    #plt.show()
    #print(pred_temp_out.shape)
    #test_pred_temp_out=den_lay(test_concat_hid_feats)
    #plt.plot(test_pred_temp_out[0])
    #plt.show()
    #droopout=layers.Dropout(d_rate)
    #pred_temp_outd=droopout(pred_temp_out)
    #den_lay2=layers.Dense(dense_widths[1],activation='elu')
    #den_lay2_out=den_lay2(pred_temp_outd)
    #pred_lay=layers.Dense(dense_widths[2],activation='elu')
    #non_norm_pred=pred_lay(den_lay2_out)
    #test_non_norm_pred=pred_lay(test_pred_temp_out)
    #plt.plot(test_non_norm_pred[0])
    #plt.show()
    #norm_pred=layers.Softmax()(non_norm_pred)
    #test_norm_pred=layers.Softmax()(test_non_norm_pred)
    #plt.plot(test_norm_pred[0])
    #plt.show()
    #build model
    #model = keras.Model(inputs=[input_atom_adjs,input_atom_feats,input_bond_adjs,input_bond_feats], outputs=norm_pred)
    #exit()
    model = keras.Model(inputs=[input_atom_adjs,input_atom_feats], outputs=pred_temp_out)
    model.compile(
            loss=tanimoto_loss,#sigmoid_cross_entropy_with_logits_loss,#keras.losses.MeanSquaredError(),#crossentropy,#fp_tp_loss,#"BinaryCrossentropy",#keras.losses.MeanSquaredError(),
            optimizer=keras.optimizers.Adam(learning_rate=learn_r, beta_1=0.8, beta_2=0.999),#keras.optimizers.Adadelta(learning_rate=learn_r),#
            
            #optimizer=keras.optimizers.Adam(learning_rate=learn_r,beta_1=0.9, beta_2=0.999,epsilon=1e-7),#beta_1=0.9, beta_2=0.999, epsilon=1e-07
            metrics=["MeanSquaredError"],
        )
    model_fpt=keras.Model(inputs=[input_atom_adjs,input_atom_feats],outputs=atom_gcn_out)
    model_fpt.compile(
            loss=sigmoid_cross_entropy_with_logits_loss,#fp_tp_loss,#"BinaryCrossentropy",#keras.losses.MeanSquaredError(),
            optimizer=keras.optimizers.Adam(learning_rate=learn_r, beta_1=0.8, beta_2=0.999),#keras.optimizers.Adadelta(learning_rate=learn_r),#
            
        )
        
    model.summary()

    # model.fit([all_adj_mat,all_nod_mat,all_lin_mat,all_edg_mat], all_mol_act,epochs=epochs,batch_size=batch,validation_split=v_split,)
    model.fit(train_dataset,epochs=epochs,validation_data=test_dataset,validation_freq=1,callbacks=[tensorboard_callback,tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)])#,batch_size=batch)#,validation_split=v_split,)
    model.save("trained_models/new_model_fptmap_"+str(fpt_type)+"_"+str(model_i)+".tf")
    model_fpt.save("trained_models/new_model_fptmap_"+str(fpt_type)+"_"+str(model_i)+"_fpt.tf")
    np.save("trained_models/new_model_fptmap_"+str(fpt_type)+"_"+str(model_i)+"_config",model_config)


    #val_start=int(all_mol_act.shape[0]*(1.-v_split))
    #predict_EC50=model.predict([all_adj_mat[val_start:],all_nod_mat[vasl_start:],all_lin_mat[val_start:],all_edg_mat[val_start:]],batch_size=batch)
    #predict_EC50=model.predict(test_dataset)
    #tanimoto_scores=[]
    #for f, b in zip(test_dataset, predict_EC50):
    #    print(f[-1])
    #    print(b)
    #    tanimoto_scores.append(tanimoto_score(f[-1],b))
    #plt.hist(tanimoto_scores,bins=100)
    #plt.xlabel("tanimoto score")
    #plt.ylabel("counts")
    #plt.show()
    #gen_roc(predict_EC50,valid_output,[x for x in np.linspace(0.00,1,20)])
    #plt.hist(predict_EC50[:,0],alpha=0.5,bins=50)
    ##plt.hist(predict_EC50[:,1],alpha=0.5)
    #plt.show()
    #exit()

#modeli
#0    500 epochs w/ dilations