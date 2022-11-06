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

def load_numpy_tot(tmod,name):
    indices=np.load("data_sets/"+str(name)+"_i_"+str(tmod)+".npy")
    values=np.load("data_sets/"+str(name)+"_v_"+str(tmod)+".npy")
    shape=np.load("data_sets/"+str(name)+"_s_"+str(tmod)+".npy")
    sparse_dat=tf.SparseTensor(indices=indices,
                  values=values,
                  dense_shape=shape)
    return sparse_dat
def gen_roc(predicted,truth,cutts):
    print("computing ROC")
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
    #plt.plot(false_poss,true_poss,marker='o')
    ##axes[1].plot(false_negss,true_negss,marker='o')
    #plt.plot([0,1],[0,1],color='k')
    #for i in range(len(cutts)):
    #    plt.annotate(str(cutts[i])[0:4],[false_poss[i]-.1,true_poss[i]])
    ##axes[1].plot([0,1],[0,1],color='k')
    #plt.xlabel("false positive rate")
    #plt.ylabel("true positive rate")
    ##axes[1].set_xlabel("false negative rate")
    ##axes[1].set_ylabel("true negative rate")
    #plt.show()
    #np.save("trained_models/roc_"+str(model_i)+"_dat",np.array([false_poss,true_poss]))
    np.save(model_config["out_dir"]+"/roc_dat_"+str(fpt_type)+"_"+shuf_i,[false_poss,true_poss])

def crossentropy(y_true,y_pred):
    return tf.reduce_mean(-1.*y_true[:,0]*tf.math.log(tf.clip_by_value(y_pred[:,0],1e-10,1.)) - (y_true[:,1])*tf.math.log(tf.clip_by_value(y_pred[:,1],1e-10,1.)))


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

fpt_type=sys.argv[1]#"cicular6"
shuf_i=sys.argv[2]
fpt_sizes={}
fpt_sizes["rdkfs"]=2048
fpt_sizes["maccs"]=167
fpt_sizes["cicular4"]=1024
fpt_sizes["cicular6"]=1024
fpt_sizes["cicular8"]=1024
fpt_sizes["circularfeat4"]=1024
fpt_sizes["circularfeat6"]=1024
fpt_sizes["circularfeat8"]=1024


model_config={}
model_config["out_dir"]="cicular_4_models_6_3_22"
#old out dir was "cicular_4_models_6_17_21"
model_config["shuf_i"]=shuf_i
res_size=2048
model_config["res_size"]=res_size

glob_rseed=18746
tf.random.set_seed(glob_rseed)
model_config["glob_rseed"]=glob_rseed

loca_rseed=512
model_config["loca_rseed"]=loca_rseed

model_config["fpt_type"]=fpt_type

v_split=0.3
model_config["v_split"]=v_split

n_dense=2
model_config["n_dense"]=n_dense

dense_widths=[int(x) for x in np.linspace(fpt_sizes[fpt_type],2,n_dense+1)]#[int(x) for x in np.linspace(200*(g_conv_filts[-1]),2,n_dense+1)]
model_config["dense_widths"]=dense_widths

dense_act='elu'
model_config["dense_act"]=dense_act

epochs=1000
model_config["epochs"]=epochs

batch=100#1000
model_config["batch"]=batch

#pre_fetch=10000
learn_r=2e-5#5e-5
model_config["learn_r"]=learn_r

d_rate=.25
model_config["d_rate"]=d_rate


retrain=1

tmod=1
model_config["tmod"]=tmod


model_i=int(sys.argv[3])
model_config["model_i"]=model_i


print("Loading Data")
#fpts_dats,all_mol_act=load_the_data(1,fpt_type)
#n_dat=all_mol_act.shape[0]
#full_dataset=tf.data.Dataset.from_tensor_slices((fpts_dats,all_mol_act)).shuffle(buffer_size=n_dat,seed=loca_rseed)
#train_dataset=full_dataset.take(int((1-v_split)*n_dat))
#test_dataset=full_dataset.skip(int((1-v_split)*n_dat))
##train_dataset = tf.data.Dataset.from_tensor_slices((fpts_dats[:int((1-v_split)*n_dat)],all_mol_act[:int((1-v_split)*n_dat)])).shuffle(buffer_size=n_dat)#,seed=loca_rseed)
##test_dataset = tf.data.Dataset.from_tensor_slices((fpts_dats[int((1-v_split)*n_dat):],all_mol_act[int((1-v_split)*n_dat):])).shuffle(buffer_size=n_dat)#,seed=loca_rseed)
#
#tf.data.experimental.save(train_dataset,"data_sets/new_data_6_13_21/train_dataset_"+fpt_type+"_"+str(tmod))
#tf.data.experimental.save(test_dataset,"data_sets/new_data_6_13_21/test_dataset_"+fpt_type+"_"+str(tmod))
#exit()
#
#all_mol_act=load_numpy_tot(tmod,"act_mat")
#all_mol_act=tf.sparse.to_dense(all_mol_act)
#print("done")
##
#n_dat=all_mol_act.shape[0]
#

##Uncomment to remake tf datasets
#out_fpt=[]
#n_per=10000
#n_set=63
#
#all_mol_act=[]
#for i in range(0,n_set*n_per,n_per):
#    out_fpt.append(np.load("data_sets/GLASS_6_2_21basic_fpt_"+fpt_type+"_"+str(i)+".npy"))
#    acts=np.load("data_sets/GLASS_6_2_21mol_act_"+str(i)+".npy")
#    
#    all_mol_act.extend([np.array([[1.,0.]]) if x < 1e3 else np.array([[0.,1.]]) for x in acts])
#
#
#n_per=10000
#n_set=8
#for i in range(0,n_set*n_per,n_per):
#    out_fpt.append(np.load("data_sets/DUDE_6_2_21basic_fpt_"+fpt_type+"_"+str(i)+".npy"))
#    acts=np.load("data_sets/DUDE_6_2_21mol_act_"+str(i)+".npy")
#
#    all_mol_act.extend([np.array([[1.,0.]]) if x < 1e3 else np.array([[0.,1.]]) for x in acts])
#out_fpt=np.concatenate(out_fpt,axis=0).astype(np.float32)
#all_mol_act=np.concatenate(all_mol_act,axis=0).astype(np.float32)
#n_dat=all_mol_act.shape[0]
#print(n_dat)
#
#train_dataset = tf.data.Dataset.from_tensor_slices((out_fpt[:int((1-v_split)*n_dat)],all_mol_act[:int((1-v_split)*n_dat)])).shuffle(buffer_size=n_dat,seed=loca_rseed)
#test_dataset = tf.data.Dataset.from_tensor_slices((out_fpt[int((1-v_split)*n_dat):],all_mol_act[int((1-v_split)*n_dat):])).shuffle(buffer_size=n_dat,seed=loca_rseed)
#for i,data in enumerate(test_dataset):
#   if i <=8:
#       print(data[-1])
#   else:
#       break
#
#tf.data.experimental.save(train_dataset,"data_sets/train_dataset_"+fpt_type+"_"+str(tmod))
#tf.data.experimental.save(test_dataset,"data_sets/test_dataset_"+fpt_type+"_"+str(tmod))
#exit()

train_dataset=tf.data.experimental.load("data_sets/new_data_6_13_21/train_dataset_"+fpt_type+"_"+str(tmod)+"_"+shuf_i,element_spec =(tf.TensorSpec(shape=( fpt_sizes[fpt_type]), dtype=tf.float32, name=None), tf.TensorSpec(shape=( 2), dtype=tf.float32, name=None)))
test_dataset=tf.data.experimental.load("data_sets/new_data_6_13_21/test_dataset_"+fpt_type+"_"+str(tmod)+"_"+shuf_i,element_spec =(tf.TensorSpec(shape=( fpt_sizes[fpt_type]), dtype=tf.float32, name=None), tf.TensorSpec(shape=( 2), dtype=tf.float32, name=None)))

n_dat=tf.data.experimental.cardinality(train_dataset).numpy()+tf.data.experimental.cardinality(test_dataset).numpy()
print(tf.data.experimental.cardinality(train_dataset).numpy(),tf.data.experimental.cardinality(test_dataset).numpy())
train_dataset=train_dataset.batch(batch)
test_dataset=test_dataset.batch(batch)
valid_output=[]
for data in test_dataset:
    valid_output.append(data[-1])
valid_output=tf.concat(valid_output,axis=0)


if retrain==1:
    mirrored_strategy = tf.distribute.MirroredStrategy()
    with mirrored_strategy.scope():
    
        input_fpts = keras.Input(shape=(fpt_sizes[fpt_type]))
        resevoir=layers.Dense(res_size,activation='linear',use_bias=False,trainable=False)
        res_out=resevoir(input_fpts)
        dense_layers=[]
        dense_outs=[]
        for i in range(n_dense):
            
            if i ==0:
                dense_layers.append(layers.Dense(dense_widths[i+1],activation=dense_act))##, kernel_regularizer='l2'))
                dense_outs.append(dense_layers[-1](res_out))
            else:
                dense_layers.append(layers.Dropout(d_rate))
                dense_outs.append(dense_layers[-1](dense_outs[-1]))
                dense_layers.append(layers.Dense(dense_widths[i+1],activation=dense_act))
                dense_outs.append(dense_layers[-1](dense_outs[-1]))
                
        #ec50_outt=layers.Dense(1,activation='linear')(dense_outs[-1])
        #ec50_out=layers.Dense(1,activation='sigmoid')(ec50_outt)
        ec50_out=layers.Softmax()(dense_outs[-1])
        #ec50_out=tf.reshape(ec50_out,(tf.shape(ec50_out)[0],1,2))
        log_dir = model_config["out_dir"]+"/t_board_"+str(fpt_type)+"_"+shuf_i
        if os.path.isdir(log_dir):
            shutil.rmtree(log_dir)
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=0)
        model = keras.Model(inputs=input_fpts, outputs=ec50_out)
        model.compile(
            loss=crossentropy,#keras.losses.MeanSquaredError(),
            optimizer=keras.optimizers.Adam(learning_rate=learn_r),
            #metrics=[],#"MeanSquaredError"
        )
    
    model.summary()
    #tf.keras.utils.plot_model(model, to_file='model_noseq3.png')

    model.fit(train_dataset,validation_data=test_dataset,epochs=epochs,callbacks=[tensorboard_callback,tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)])
    #np.save("trained_models/model_"+str(model_i)+"_perme",perme)
    model.save(model_config["out_dir"]+"/model_"+str(fpt_type)+"_"+shuf_i+".tf")
    np.save(model_config["out_dir"]+"/model_"+str(fpt_type)+"_"+shuf_i+"_config",model_config)
    #exit()
else:
    #mirrored_strategy = tf.distribute.MirroredStrategy()
    #with mirrored_strategy.scope():
    model=keras.models.load_model(model_config["out_dir"]+"/model_"+str(fpt_type)+"_"+shuf_i+".tf")
    model.compile(
            loss=crossentropy,#keras.losses.MeanSquaredError(),
            optimizer=keras.optimizers.Adam(learning_rate=learn_r),
            metrics=["MeanSquaredError"],
        )

predict_EC50=model.predict(test_dataset,batch_size=batch)
gen_roc(predict_EC50,valid_output,[x for x in np.linspace(0.00,1,20)])