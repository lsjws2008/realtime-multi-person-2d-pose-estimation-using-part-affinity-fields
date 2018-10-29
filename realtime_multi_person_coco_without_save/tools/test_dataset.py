#!/usr/bin/env python2
# -*- coding: utf-8 -*-
from PIL import Image, ImageDraw
import numpy as np
import os

lisp = os.listdir('F:/trains/')
a = '000000131697'
imgs = Image.open('F:trains/'+a+'/img.jpg')
draw = ImageDraw.Draw(imgs)
draw.ellipse((90 - 3, 170 - 3, 90 +3, 170 + 3), fill = 'blue', outline ='blue')#90 170 54 217
"""
390 200 321 252
268 149 223 182
90 170 54 217
405 15 389 54
"""
draw.ellipse((54 - 3, 217 - 3, 54 +3, 217 + 3), fill = 'blue', outline ='blue')#90 170 54 217

draw.ellipse((390 - 3, 200 - 3, 390 +3, 200 + 3), fill = 'blue', outline ='blue')
draw.ellipse((321 - 3, 252 - 3, 321 +3, 252 + 3), fill = 'blue', outline ='blue')

draw.ellipse((268 - 3, 149 - 3, 268 +3, 149 + 3), fill = 'blue', outline ='blue')
draw.ellipse((223 - 3, 182 - 3, 223 +3, 182 + 3), fill = 'blue', outline ='blue')

draw.ellipse((405 - 3, 15 - 3, 405 +3, 15 + 3), fill = 'blue', outline ='blue')
draw.ellipse((389 - 3, 54 - 3, 389 +3, 54 + 3), fill = 'blue', outline ='blue')
imgs.save('test_out_img/img.jpg')
for i in range(12):
    confidence_map = np.load('F:trains/'+a+'/s_lab.npy')
    imgs = Image.open('F:trains/'+a+'/img.jpg')
    print(imgs.size)
    map_0 = confidence_map[:, :, i]
    map_0 = np.expand_dims(map_0, 2)
    map_0 = map_0*255
    map_0_to = np.concatenate((map_0, map_0), 2)
    map_0_to = np.concatenate((map_0_to, map_0), 2)
    map_0_to = Image.fromarray(np.uint8(map_0_to)).resize((imgs.size[1], imgs.size[0]))
    imgs = np.transpose(np.array(imgs), (1, 0, 2))
    imgs = imgs+np.array(map_0_to)
    imgs[imgs>255] = 255
    
    im = Image.fromarray(np.uint8(map_0_to).transpose((1, 0, 2)))
    #im.show()
    im.save('test_out_img/'+str(i)+'.jpg')
    print(imgs.shape)
    imgs = np.clip(imgs, 0, 255)
    imgs = np.transpose(imgs, [1,0,2])
    Image.fromarray(np.uint8(imgs)).save('test_out_img/'+str(i+17)+'.jpg')
"""
confidence_map = np.load('F:/trains/'+a+'/l_lab.npy')
imgs = Image.open('F:/trains/'+a+'/img.jpg')
imgs.show()
imgs = imgs.resize((46,46))
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
"""