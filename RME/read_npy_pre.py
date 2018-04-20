# -*- coding: utf-8 -*-
"""
Created on Fri Apr 13 23:58:34 2018

@author: johnny
"""
import numpy as np

D = np.load("C:/Users/johnny/Downloads/vgg19.npy", encoding = 'latin1')
print(D.item().get('conv5_1')[0].shape)
print(D.iterkeys())
