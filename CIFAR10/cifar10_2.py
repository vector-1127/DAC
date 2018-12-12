#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov  5 16:46:33 2016

@author: changjianlong
"""

from __future__ import print_function
import os,sys
os.environ['KERAS_BACKEND'] = 'theano'
os.environ['THEANO_FLAGS']='device=gpu1,lib.cnmem=0.45,mode=FAST_RUN,floatX=float32,optimizer=fast_compile'

#==============================================================================
# import os
# os.environ['KERAS_BACKEND'] = 'tensorflow'
# os.environ['CUDA_VISIBLE_DEVICES'] = '6'
# os.environ['IMAGE_DIM_ORDERING'] = 'tf'
# import tensorflow as tf
# config = tf.ConfigProto()
# config.gpu_options.per_process_gpu_memory_fraction = 0.3
# tf.Session(config=config)
#==============================================================================
###STL good
import numpy as np
import h5py
from keras.datasets import mnist,cifar10
from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Flatten, Input
from keras.layers import Dropout, merge, Lambda, Reshape, Convolution2D, MaxPooling2D
from keras.layers import BatchNormalization, AveragePooling2D, Highway,Bidirectional
from keras.utils import np_utils
from keras import backend as K
from keras.engine.topology import Layer
import matplotlib.pyplot as plt
import scipy.io as sio
import json
from keras.models import model_from_json
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import RMSprop
from keras.layers.advanced_activations import LeakyReLU
from keras.models import load_model
from sys import path
path.append('/home/changjianlong/clustering')
from myMetrics import *

global upper, lower
datagen = ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.09,
    height_shift_range=0.09,
    channel_shift_range=0.05,
    horizontal_flip=True,
    rescale=0.975,
    zoom_range=[0.95,1.05]
)

class Adaptive(Layer):
    def __init__(self, norm=2.0, learnable = False, **kwargs):
        self.norm = norm
        self.learnable = learnable
        super(Adaptive, self).__init__(**kwargs)
        
    def build(self, input_shape):
        self.nb_sample = input_shape[0]
        self.nb_dim = input_shape[1]
        self.learn_norm = K.variable(self.norm)
        if self.learnable == True:
            self.trainable_weights = [self.learn_norm]
        else:
            self.non_trainable_weights = [self.learn_norm]
        
    def call(self, x, mask = None):
        y = self.transfer(x)
        return y
        
    def get_output_shape_for(self, input_shape):
        return (input_shape[0],input_shape[1])
        
    def transfer(self, x):
        y = K.pow(K.sum((x+K.epsilon())**self.learn_norm, axis = 1), 1./self.learn_norm)
        y = K.expand_dims(y, dim = 1)
        y = K.repeat_elements(y, self.nb_dim, axis = 1)
        return x/y

class DotDist(Layer):
    def __init__(self, **kwargs):
        super(DotDist, self).__init__(**kwargs)
        
    def call(self, x, mask = None):
        d = K.dot(x, K.transpose(x))
        return d
        
    def get_output_shape_for(self, input_shape):
        return (input_shape[0],input_shape[0])

class CosineDist(Layer):
    def __init__(self, **kwargs):
        super(CosineDist, self).__init__(**kwargs)
        
    def call(self, x, mask = None):
        x = K.l2_normalize(x, axis =1)
        d = K.dot(x, K.transpose(x))
        return d
        
    def get_output_shape_for(self, input_shape):
        return (input_shape[0],input_shape[0])

#data
nb_classes = 10
img_channels, img_rows, img_cols = 3, 32, 32
file = h5py.File('/home/changjianlong/datasets/CIFAR10.h5')
X_train = file['X_train'][:]
y_true = file['y_train'][:]
y_true = y_true.astype('int64')
file.close()
index = np.arange(X_train.shape[0])
np.random.shuffle(index)
X_train = X_train[index]
y_true = y_true[index]
tempmap = np.copy(index)
#parameters
batch_size = 32
epoch = 50
nb_epoch = 10
upper = 0.99
lower = 0.8
eta = (upper-lower)/epoch/2
nb = 1000
run = 'Clustering'
#run = 'Classification'
nb_samples = [500]
print(run)
#model
#==============================================================================
# X_train = np.transpose(X_train, (0,2,3,1))
#==============================================================================
inp_ = Input(shape=(img_channels, img_rows, img_cols))
x = Convolution2D(64,3,3,init = 'he_normal')(inp_)
x = BatchNormalization(mode=2,axis = 1)(x)
x = Activation('relu')(x)
x = Convolution2D(64,3,3,init = 'he_normal')(x)
x = BatchNormalization(mode=2,axis = 1)(x)
x = Activation('relu')(x)
x = Convolution2D(64,3,3,init = 'he_normal')(x)
x = BatchNormalization(mode=2,axis = 1)(x)
x = Activation('relu')(x)
x = MaxPooling2D(pool_size = (2,2))(x)
x = Convolution2D(128,3,3,init = 'he_normal')(x)
x = BatchNormalization(mode=2,axis = 1)(x)
x = Activation('relu')(x)
x = Convolution2D(128,3,3,init = 'he_normal')(x)
x = BatchNormalization(mode=2,axis = 1)(x)
x = Activation('relu')(x)
x = Convolution2D(128,3,3,init = 'he_normal')(x)
x = BatchNormalization(mode=2,axis = 1)(x)
x = Activation('relu')(x)
x = MaxPooling2D(pool_size = (2,2))(x)
x = Convolution2D(10, 1, 1,init = 'he_normal')(x)
x = BatchNormalization(mode = 2,axis = 1)(x)
x = Activation('relu')(x)
x = AveragePooling2D(pool_size = (3,3))(x)
x0 = Flatten()(x)
x1 = Dense(10, init='identity')(x0)
x1 = BatchNormalization(mode = 2)(x1)
x1 = Activation('relu')(x1)
x1 = Dense(10, init='identity')(x1)
x1 = BatchNormalization(mode = 2)(x1)
y = Activation('softmax', name = 'aux_output')(x1)
my_l2_norm = Adaptive(norm = 2)(y)
z = Adaptive(norm = 2)(y)
dotdist = DotDist(name = 'main_output')(z)

norm_l2 = Model(input=[inp_], output=[x0])
cluster_l1 = Model(input=[inp_], output=[y])
cluster_l2 = Model(input=[inp_], output=[my_l2_norm])
cluster_l = Model(input=[inp_], output=[z])
model = Model(input=[inp_], output=[y, dotdist])


location = str(sys.argv[0])
location = location.replace('.py','.h5')
weight_path = location.replace('.h5','')
print(location)
print(weight_path)
if not os.path.exists(weight_path):
    os.makedirs(weight_path)

if run == 'Clustering':
    acc = []
    ari = []
    nmi = []
    ind = []
    output = []
    
    Output = cluster_l1.predict(X_train)
    y_pred = np.argmax(Output,axis = 1)
    nmit = NMI(y_true,y_pred)
    arit = ARI(y_true,y_pred)
    acct, indt = ACC(y_true,y_pred)
    model_weight_location = location.replace('.h5','/model_weight_epoch_{}.h5'.format(0))
    cluster_l1.save_weights(model_weight_location)
    print(nmit, arit, acct)
    acc.append(acct)
    ari.append(arit)
    nmi.append(nmit)
    ind.append(indt)
    output.append(Output)
    
    index = np.arange(X_train.shape[0])
    index_loc = np.arange(nb)
    for e in range(epoch):
        np.random.shuffle(index)
        print(e,lower, upper)
        def my_binary_crossentropy(y_true, y_pred):
            w1 = K.switch(y_true<lower,1,0)
            w2 = K.switch(y_true>upper,1,0)
            w = w1 + w2
            nb_selected = K.sum(w)
            y_true = K.switch(y_true>(upper+lower)/2,1,0)
            return K.sum(w*K.binary_crossentropy(y_true, y_pred), axis=-1)/nb_selected

        model.compile(optimizer=RMSprop(0.001), 
              loss={'main_output':my_binary_crossentropy,'aux_output':'binary_crossentropy'}, 
              loss_weights={'main_output':1,'aux_output':1})
        
        if X_train.shape[0]>nb:
            for i in range(X_train.shape[0]//nb):
                Xbatch = X_train[index[np.arange(i*nb,(i+1)*nb)]]
                Y = cluster_l2.predict(Xbatch)
                Y_l1 = cluster_l1.predict(Xbatch)
                Ybatch = np.dot(Y,Y.T)
                for k in range(nb_epoch):
                    np.random.shuffle(index_loc)
                    for j in range(Xbatch.shape[0]//batch_size):
                        address = index_loc[np.arange(j*batch_size,(j+1)*batch_size)]
                        X_batch = Xbatch[address]
                        Y_batch = Ybatch[address,:][:,address]
                        Y_batch_l1 = Y_l1[address]
                        sign = 0
                        for X_batch_i in datagen.flow(X_batch, batch_size=batch_size,shuffle=False):
                            loss = model.train_on_batch([X_batch_i],[Y_batch_l1, Y_batch])
                            sign += 1
                            if sign>1:
                                break
                if i%10==0:
                    print('Epoch: %d, batch: %d/%d, loss: %f, loss1: %f, loss2: %f, nb1: %f'%
                          (e+1,i+1,X_train.shape[0]//nb,loss[0],loss[1],loss[2],np.mean(Ybatch)))
        else:
            print('error')
        upper = upper - eta
        lower = lower + eta
        Output = cluster_l1.predict(X_train)
        y_pred = np.argmax(Output,axis = 1)
        nmit = NMI(y_true,y_pred)
        arit = ARI(y_true,y_pred)
        acct, indt = ACC(y_true,y_pred)
        model_weight_location = location.replace('.h5','/model_weight_epoch_{}.h5'.format(e+1))
        cluster_l1.save_weights(model_weight_location)
        print(nmit, arit, acct)
        acc.append(acct)
        ari.append(arit)
        nmi.append(nmit)
        ind.append(indt)
        output.append(Output)
        
        if os.path.exists(location):
            os.remove(location)
        file = h5py.File(location,'w')
        file.create_dataset('acc',data = acc)
        file.create_dataset('nmi',data = nmi)
        file.create_dataset('ari',data = ari)
        file.create_dataset('ind',data = ind)
        file.create_dataset('output',data = output)
        file.create_dataset('tempmap',data = tempmap)
        file.close()
else:
    for nb_sample in nb_samples:
        X = X_train[0:nb_sample]
        Y = np_utils.to_categorical(y_true[0:nb_sample], nb_classes)
        X_test = X_train[-1000:-1]
        Y_test = np_utils.to_categorical(y_true[-1000:-1], nb_classes)
        cluster_l1.fit(X, Y, nb_epoch=100,validation_data=(X_test, Y_test), verbose=2)
#==============================================================================
#         cluster_l1.fit_generator(datagen.flow(X, Y, batch_size=32), 
#                               nb_sample // 32,nb_epoch=100, verbose=2, validation_data=(X_test, Y_test))
#==============================================================================
    model_weight_location = location.replace('.h5','/model_weight_epoch_{}.h5'.format(e))
    cluster_l1.load_weights(model_weight_location)
    f = h5py.File(location)
    acc = f['acc'][:]
    nmi = f['nmi'][:]
    ari = f['ari'][:]
    ind = f['ind'][:]
    output = f['output'][:]
    tempmap = f['tempmap'][:]
    f.close()
    Output = cluster_l1.predict(X_train)
    y_pred = np.argmax(Output,axis = 1)
    acc, ind = ACC(y_true,y_pred)
    
    y_trasfer = np.ones_like(y_true)
    for i in range(nb_classes):
        index = np.where(y_true == i)
        indext = np.where(ind[:, 1]==i)
        y_trasfer[index] = ind[indext, 0]
    acct, indt = ACC(y_trasfer,y_pred)
    assert(acc==acct)
    
    for nb_sample in nb_samples:
        X = X_train[0:nb_sample]
        Y = np_utils.to_categorical(y_trasfer[0:nb_sample], nb_classes)
        X_test = X_train[-1000:-1]
        Y_test = np_utils.to_categorical(y_trasfer[-1000:-1], nb_classes)
        cluster_l1.fit(X, Y, nb_epoch=100,validation_data=(X_test, Y_test), verbose=2)
#==============================================================================
#         cluster_l1.fit_generator(datagen.flow(X, Y, batch_size=32), 
#                               nb_sample // 32,nb_epoch=100, verbose=2,validation_data=(X_test, Y_test))
#==============================================================================







