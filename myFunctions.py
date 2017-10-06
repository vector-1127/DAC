# -*- coding: utf-8 -*-
"""
Created on Thu Oct 13 10:16:32 2016

@author: changjianlong
"""

from __future__ import print_function
import numpy as np

from keras.datasets import mnist
from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Flatten, Input, Dropout, merge
from keras.layers import Convolution2D, MaxPooling2D, BatchNormalization, AveragePooling2D
from keras.utils import np_utils
from keras import backend as K
from keras.engine.topology import Layer
import theano.tensor as T
import matplotlib.pyplot as plt
import scipy.io as sio
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import RMSprop


def myInit(data,nb_iter = 0,th = 0.95,std = 2000):
    nb_data,nb_feature = data.shape
    
    x = K.placeholder(ndim = 2)
    vecProd = K.dot(x,x.T)
    sumSq1 = K.sum(x**2,axis = 1)
    sumSq1 = K.expand_dims(sumSq1,dim = 1)
    sumSq1 = K.repeat_elements(sumSq1,nb_data,axis = 1)
    eulerDist = sumSq1 + sumSq1.T - 2*vecProd
    relat = K.switch(K.exp(-1*eulerDist/std)>th,1,0)
    
    rela = relat
    temp = K.sign(K.dot(rela,relat))
    k = 0
    while K.not_equal(K.sum(K.equal(temp,rela)),nb_data*nb_data) and k<nb_iter:
        rela = temp
        temp = K.sign(K.dot(rela,relat))
        k += 1
    
    f = K.function([x],[rela])
    return f([data])[0]




