#!/usr/bin/env python2
# -*- coding: utf-8 -*-
from PIL import Image
import numpy as np
import os

lisp = os.listdir('F:/trains/')
for i in range(17):
    confidence_map = np.load('F:trains/000000537548/s_lab.npy')
    imgs = Image.open('F:trains/000000537548/img.jpg').resize((100,50))
    imgs = np.transpose(np.array(imgs), (1, 0, 2))

    map_0 = confidence_map[:, :, i]
    map_0[map_0 < 0.5] = 0
    map_0 = np.expand_dims(map_0, 2)
    map_0 = map_0*255
    map_0_to = np.concatenate((map_0, map_0), 2)
    map_0_to = np.concatenate((map_0_to, map_0), 2)
    imgs = imgs+map_0_to
    imgs[imgs>255] = 0
    
    im = Image.fromarray(np.uint8(map_0_to).transpose((1, 0, 2)))
    #im.show()
    im.save('test_out_img/'+str(i)+'.jpg')
    print(imgs.shape)
    imgs = np.clip(imgs, 0, 255)
    imgs = np.transpose(imgs, [1,0,2])
    Image.fromarray(np.uint8(imgs)).save('test_out_img/'+str(i+17)+'.jpg')
confidence_map = np.load('F:/trains/000000537548/l_lab.npy')
imgs = Image.open('F:/trains/000000537548/img.jpg')
imgs.show()
imgs = imgs.resize((100,50))
imgs = np.transpose(np.array(imgs), (1, 0, 2))
    
for i in range(12):
    map_0 = confidence_map[:, :, i*2]
    map_0[map_0 != 0]=1
    if np.sum(map_0)!=0:
        print(i+34)
    map_0 = np.expand_dims(map_0, 2)
    map_0 = map_0*255
    map_0_to = np.concatenate((map_0, map_0), 2)
    map_0_to = np.concatenate((map_0_to, map_0), 2)
    img = imgs+map_0_to
    img[img>255] = 0
    img[img<0] = 0
    
    im = Image.fromarray(np.uint8(map_0_to).transpose((1, 0, 2)))
    im.save('test_out_img/'+str(i+34)+'.jpg')
    im = Image.fromarray(np.uint8(img).transpose((1, 0, 2)))
    im.save('test_out_img/'+str(i+51)+'.jpg')