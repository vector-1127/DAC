# -*- coding: utf-8 -*-

import numpy as np
import h5py
import matplotlib.pyplot as plt
from skimage import transform
import os

# save data
address = "../datasets/imagenet10/"
size = 96
data = np.zeros((13000,3,size,size))
label = np.zeros((13000,))
c = 0
k = 0
for i in os.listdir(address):
    for j in os.listdir(address+i):
        img = np.double(plt.imread(address+i+'/'+j))/255
        if len(img.shape)==3:
            data[1300*c+k] = transform.resize(np.transpose(img,(2,0,1)),(3,size,size))
        else:
            temp = np.zeros((3,size,size))
            temp1 = transform.resize(img,(size,size))
            temp[0]=temp1;temp[1]=temp1;temp[2]=temp1
            data[1300*c+k] = temp
        label[1300*c+k] = c
        k += 1
    c += 1
    k = 0

data[:,0,:,:] -= np.mean(data[:,0,:,:])
data[:,1,:,:] -= np.mean(data[:,1,:,:])
data[:,2,:,:] -= np.mean(data[:,2,:,:])

if os.path.exists('ImageNet10.h5'):
        os.remove('ImageNet10.h5')
file = h5py.File('ImageNet10.h5','w')
file.create_dataset('X_train',data = data)
file.create_dataset('y_train',data = label)
file.close()

# load data
file = h5py.File('ImageNet10.h5')
X_train = file['X_train'][:]
y_true = file['y_train'][:]
y_true = y_true.astype('int64')
file.close()