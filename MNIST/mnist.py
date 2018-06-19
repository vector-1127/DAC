# -*- coding: utf-8 -*-
"""
Created on Thu Oct 13 17:14:45 2016

@author: changjianlong
"""

from __future__ import print_function
import os,sys
os.environ['KERAS_BACKEND'] = 'theano'
os.environ['THEANO_FLAGS']='device=gpu3,lib.cnmem=1,mode=FAST_RUN,floatX=float32,optimizer=fast_compile'

import numpy as np
#np.random.seed(1337)  # for reproducibility
import matplotlib.pyplot as plt
import scipy.io as sio
import h5py
from keras.datasets import mnist,cifar10
from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Flatten, Input, Dropout, merge, Lambda
from keras.layers import MaxPooling2D, BatchNormalization, AveragePooling2D
from keras.layers import Convolution2D, Convolution1D,AveragePooling1D
from keras.utils import np_utils
from keras import backend as K
from keras.engine.topology import Layer
from keras.preprocessing.image import ImageDataGenerator
from keras.layers.advanced_activations import LeakyReLU
from keras.regularizers import l2, activity_l2
import theano.tensor as T
import matplotlib.pyplot as plt
from keras.optimizers import RMSprop
import scipy.io as sio
from sys import path
path.append('../DAC')
from myMetrics import *
from myImage import myImageDataGenerator

datagen = myImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    channel_shift_range=0.05,
    rescale=0.975,
    zoom_range=[0.95,1.05]
)

datagen_keras = ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    channel_shift_range=0.05,
    rescale=0.975,
    zoom_range=[0.95,1.05]
)

def my_dist_output_shape(shapes):
    shape1, shape2 = shapes
    return (shape1[0], shape1[0])

def my_dist(y_pred):
    y_pred1,y_pred2 = y_pred
    norm1 = K.sqrt(K.sum(y_pred1**2,axis = 1))
    norm1 = K.reshape(norm1,(norm1.shape[0],1))
    norm1 = K.reshape(K.repeat(norm1,y_pred1.shape[1]),y_pred1.shape)
    y_pred1 = y_pred1/norm1
    
    norm2 = K.sqrt(K.sum(y_pred2**2,axis = 1))
    norm2 = K.reshape(norm2,(norm2.shape[0],1))
    norm2 = K.reshape(K.repeat(norm2,y_pred2.shape[1]),y_pred2.shape)
    y_pred2 = y_pred2/norm2
    
    return K.dot(y_pred1,y_pred2.T)

def myDist(y_pred):
    y_pred1,y_pred2,sth = y_pred
    norm1 = K.sqrt(K.sum(y_pred1**2,axis = 1))
    norm1 = K.reshape(norm1,(norm1.shape[0],1))
    norm1 = K.reshape(K.repeat(norm1,y_pred1.shape[1]),y_pred1.shape)
    y_pred1 = y_pred1/norm1
    
    norm2 = K.sqrt(K.sum(y_pred2**2,axis = 1))
    norm2 = K.reshape(norm2,(norm2.shape[0],1))
    norm2 = K.reshape(K.repeat(norm2,y_pred2.shape[1]),y_pred2.shape)
    y_pred2 = y_pred2/norm2
    
    return K.switch(K.dot(y_pred1,y_pred2.T)>sth,1,0)

def myLoss(y_true, y_pred):
    loss = K.mean((y_pred-y_true)**2)
    return loss

def mnistNetwork(inp_img):
    x = Convolution2D(64,3,3,init = 'he_normal')(inp_img)
    x = BatchNormalization(mode=2,axis = 1)(x)
    x = Activation('relu')(x)
    x = Convolution2D(64,3,3,init = 'he_normal')(x)
    x = BatchNormalization(mode=2,axis = 1)(x)
    x = Activation('relu')(x)
    x = Convolution2D(64,3,3,init = 'he_normal')(x)
    x = BatchNormalization(mode=2,axis = 1)(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size = (2,2))(x)
    x = BatchNormalization(mode=2,axis = 1)(x)
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
    x = BatchNormalization(mode=2,axis = 1)(x)
    x = Convolution2D(10, 1, 1,init = 'he_normal')(x)
    x = BatchNormalization(mode = 2,axis = 1)(x)
    x = Activation('relu')(x)
    x = AveragePooling2D(pool_size = (2,2))(x)
    x = BatchNormalization(mode=2,axis = 1)(x)
    x = Flatten()(x)
    
    
    x = Dense(nb_classes, init='identity')(x)
    x = BatchNormalization(mode = 2)(x)
    x = Activation('relu')(x)
    x = Dense(nb_classes, init='identity')(x)
    x = BatchNormalization(mode = 2)(x)
    x = Activation('relu')(x)
    
    y = Activation('softmax')(x)
    model = Model(input=inp_img, output=y)
    
    return model

def myMax(dist):
    dist1,dist2,sth = dist
    ma = K.max((dist1,dist2),axis = 0)
    ma = K.dot(ma,ma)
    return K.switch(ma>sth,1,0)

def myCenter(Xbatch,model,datagen,fmyDist,fmyMax,th):
    output = model.predict(Xbatch)
    center = fmyDist([output,output,th])[0]
    return center


#data
nb_classes = 10
img_channels, img_rows, img_cols = 1, 28, 28
(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train = X_train.reshape(X_train.shape[0], img_channels, img_rows, img_cols)
X_test = X_test.reshape(X_test.shape[0], img_channels, img_rows, img_cols)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255.
X_test /= 255.
nb = min(1000,X_train.shape[0])
X_train = np.vstack((X_train,X_test))
y_true = np.hstack((y_train,y_test))
y_true = y_true.astype('int64')


mean_image0 = np.mean(X_train[:,0,:,:])
X_train[:,0,:,:] -= mean_image0

index = np.arange(X_train.shape[0])
X_train = X_train[index]
y_true = y_true[index]
tempmap = np.copy(index)

#parameters
batch_size = 128
epoch = 20
nb_epoch = 5
th = 0.9


#model
baseNetwork = mnistNetwork(Input(shape=(img_channels, img_rows, img_cols)))

inp_img_t = Input(shape=(img_channels, img_rows, img_cols))
inp_img_f = Input(shape=(img_channels, img_rows, img_cols))

processed_t = baseNetwork(inp_img_t)
processed_f = baseNetwork(inp_img_f)
distance = Lambda(my_dist,output_shape=my_dist_output_shape)([processed_t, processed_f])
model = Model(input=[inp_img_t,inp_img_f], output=distance)
model.compile(loss=myLoss, optimizer='rmsprop')
cluster = Model(input=[inp_img_t], output=[processed_t])
cluster.compile(loss='categorical_crossentropy', optimizer=RMSprop(0.001), metrics=['accuracy'])

soutput1 = K.placeholder(shape=(batch_size,nb_classes))
soutput2 = K.placeholder(shape=(batch_size,nb_classes))
sth = K.placeholder(ndim = 0)
fmyDist = K.function([soutput1,soutput2,sth],[myDist([soutput1,soutput2,sth])])
sdist1 = K.placeholder(shape=(batch_size,batch_size))
sdist2 = K.placeholder(shape=(batch_size,batch_size))
fmyMax = K.function([sdist1,sdist2,sth],[myMax([sdist1,sdist2,sth])])

location = str(sys.argv[0])
location = location.replace('.py','.h5')
weight_path = location.replace('.h5','')
print(location)
print(weight_path)
if not os.path.exists(weight_path):
    os.makedirs(weight_path)


acc = []
ari = []
nmi = []
ind = []
output = []

Output = cluster.predict(X_train,batch_size = batch_size)
y_pred = np.argmax(Output,axis = 1)
nmit = NMI(y_true,y_pred)
arit = ARI(y_true,y_pred)
acct, indt = ACC(y_true,y_pred)
model_weight_location = location.replace('.h5','/model_weight_epoch_{}.h5'.format(0))
cluster.save_weights(model_weight_location)
print(nmit, arit, acct)
acc.append(acct)
ari.append(arit)
nmi.append(nmit)
ind.append(indt)
output.append(Output)

for e in range(epoch):
	for i in range(X_train.shape[0]//nb):
		if i%10==0:
			print('Epoch: %d, batch: %d/%d'%(e+1,i+1,X_train.shape[0]//nb))
		Xbatch = X_train[np.arange(i*nb,(i+1)*nb)]
		Ybatch = myCenter(Xbatch,cluster,datagen,fmyDist,fmyMax,th)
		model.fit_generator(datagen.flow(Xbatch,Ybatch,batch_size=batch_size),
							samples_per_epoch=len(Xbatch), nb_epoch=nb_epoch,verbose = 0)
	th = th - (0.9-0.5)/epoch
	
	Output = cluster.predict(X_train,batch_size = batch_size)
	y_pred = np.argmax(Output,axis = 1)
	nmit = NMI(y_true,y_pred)
	arit = ARI(y_true,y_pred)
	acct, indt = ACC(y_true,y_pred)
	model_weight_location = location.replace('.h5','/model_weight_epoch_{}.h5'.format(e+1))
	cluster.save_weights(model_weight_location)
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



