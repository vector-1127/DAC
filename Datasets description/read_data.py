# -*- coding: utf-8 -*-

import h5py

file = h5py.File('ImageNet96.h5')
X_train = file['X_train'][:]
y_true = file['y_train'][:]
y_true = y_true.astype('int64')
file.close()