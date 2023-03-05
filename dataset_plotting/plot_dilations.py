import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
import matplotlib as mpl
from mpl_toolkits.axes_grid1 import make_axes_locatable
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
import sys
from numba import jit
import shutil

params = {'mathtext.default': 'regular' ,"figure.autolayout":True }          
plt.rcParams.update(params)
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
            #print(k)
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
        return xs,normxs

#train_dataset= tf.data.experimental.load("../data_sets/train_dataset_"+str(1), element_spec =((tf.TensorSpec(shape=( 80, 80), dtype=tf.float32, name=None), tf.TensorSpec(shape=(80, 17), dtype=tf.float32, name=None)), tf.TensorSpec(shape=( 2), dtype=tf.float32, name=None)))
indices=np.load("../data_sets/adj_mat_i_"+str(100)+".npy")
values=np.load("../data_sets/adj_mat_v_"+str(100)+".npy")
shape=np.load("../data_sets/adj_mat_s_"+str(100)+".npy")
sparse_dat=tf.SparseTensor(indices=indices,
              values=values,
              dense_shape=shape)
dense_dat=tf.sparse.to_dense(sparse_dat)




max_at=50
max_dil=10
gd=graph_dilations(list(range(max_dil)))

plot_is=[0,1,2,max_dil-1]

fig,axes=plt.subplots(ncols=4,nrows=2)

for dat in dense_dat:
    atom_dilated_adjs=gd([dat])
    for ind,adj in enumerate(atom_dilated_adjs[0]):
        if ind in plot_is:
            i=plot_is.index(ind)
            divider = make_axes_locatable(axes[0][i])
            cax = divider.append_axes('right', size='5%', pad=0.05)
            im=axes[0][i].imshow(adj[0][:max_at,:max_at], vmin = 0, vmax = 1)
            fig.colorbar(im, cax=cax, orientation='vertical')
            if i ==0:
                axes[0][i].set_ylabel('$atom_{\it{j}}$')
                axes[1][i].set_yticks([0,20,40])
                axes[0][i].set_xticks([])
            else:
                axes[0][i].set_xticks([])
                axes[0][i].set_yticks([])
    for ind,adj in enumerate(atom_dilated_adjs[1]):
        if ind in plot_is:
            i=plot_is.index(ind)
            divider = make_axes_locatable(axes[1][i])
            cax = divider.append_axes('right', size='5%', pad=0.05)
            im=axes[1][i].imshow(adj[0][:max_at,:max_at], vmin = 0, vmax = 1)
            fig.colorbar(im, cax=cax, orientation='vertical')
            axes[1][i].set_xlabel('$atom_{\it{i}}$')
            axes[1][i].set_xticks([0,20,40])
            if i ==0:
                axes[1][i].set_ylabel('$atom_{\it{j}}$')
                axes[1][i].set_yticks([0,20,40])
            else:
                axes[1][i].set_yticks([])
    for axs in axes:
        for ax in axs:
            #cax = fig.add_axes([0.27, 0.8, 0.5, 0.05])
            #fig.colorbar(ax.get_images()[0], cax=cax, orientation='horizontal')
            ax.tick_params(direction='in')
    plt.savefig("dilation_figure.png",dpi=600,transparent=True)
    plt.show()
    exit()


    break
