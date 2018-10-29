# -*- coding: utf-8 -*-
"""
Created on Sat Sep 15 18:21:48 2018

@author: johnny
"""

from random import shuffle
import os
from PIL import Image
import numpy as np
import json
from math import ceil
#from draw import draw_graph
from time import time
import sys
pathname = os.path.dirname(sys.argv[0])
abs_path = pathname.split('/')[:-1]
sys.path.append('/'.join(abs_path))
from tools.color_map import range2color


needed_point = [0, 1, 2, 3, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]

def load_image(path, input_shape):
    img = Image.open(path)
    w, h = img.size
    img, scale = process_input_image(img, input_shape)

    return img, (w, h), scale

def process_input_image(img, input_shape):

    shape = img.size
    scale = [input_shape[1] / shape[0],
             input_shape[0] / shape[1]]
    
    img = np.array(img \
                   .resize((int(min(scale) * shape[0]),
                            int(min(scale) * shape[1])))) \
          / 255
          
    if len(img.shape) == 2:
        img = np.expand_dims(img,
                             2)
        img = np.concatenate((img,
                              img,
                              img),
                             2)
        
    img = np.transpose(img,
                       [1, 0, 2])
    
    img = pad_img(img, input_shape)

    return img, min(scale)

def pad_img(img, input_shape):
    
    if img.shape[0] < input_shape[0]:
        pad = np.zeros([input_shape[0] - img.shape[0],
                        img.shape[1],
                        img.shape[2]])
        
        img = np.concatenate([img, 
                              pad], 0)
        
    if img.shape[1] < input_shape[1]:
        pad = np.zeros([img.shape[0],
                        input_shape[1] - img.shape[1],
                        img.shape[2]])
        
        img = np.concatenate([img, 
                              pad], 1)
    return img

class generator():

    def __init__(self, image_path = 'C:/Users/johnny/Downloads/', 
                 json_path = 'F://datas.json', 
                 input_shape=(368, 368), 
                 input_target_shape=(92, 92),
                 sample_rate=0.5, 
                 score_threshold=0.3, 
                 delta_s = 1.5, 
                 generate_type='train'):
        
        if generate_type == 'train':
            
            self.input_target_shape = input_target_shape
            self.data = json.load(open(json_path))
            self.imgs = self.data[0]
            self.labs = self.data[1]
            self.index = []

        else:
            self.index = os.listdir(image_path)
            
        self.image_path = image_path

        self.input_shape = input_shape
        self.sample_rate = sample_rate
        self.score_threshold = score_threshold
        self.generate_type = generate_type
        self.delta_s = delta_s
        self.epoch = 0

    def compute_sample_area(self, s):
    
        s[s < self.score_threshold] = 0
        s[s >= self.score_threshold] = 1
        
        for i in range(s.shape[2]):
            
            count = np.count_nonzero(s[:, :, i])
            count = int(count * self.sample_rate)

            indexs = np.array(np.where(s[:, :, i] == 0))
            length = indexs.shape[1]

            indexs = indexs[:,
                            np.random.permutation(length)]
            indexs = indexs[:,
                            :count]

            s[indexs[0, :],
              indexs[1, :], 
              i] = 1
        
        return s

    def compute_target(self, img, lab, size, scale):

        scale = [scale*(self.input_target_shape[0]/self.input_shape[0]), 
                 scale*(self.input_target_shape[1]/self.input_shape[1])]
        
        img_x, img_y = size
        img_size = [int(scale[0]*img_x), 
                    int(scale[1]*img_y)]
        
        for point in range(17):
            if point not in needed_point:
                continue
            point_os = []
            
            for psi in lab:
                if psi[point * 3 - 1] == 2:
                    psit = np.array([psi[point * 3 - 3], 
                                     psi[point * 3 - 2]])
                    psit[0] *= scale[0]
                    psit[1] *= scale[1]
                    point_os.append(psit)
                else:
                    point_os.append(np.array([1e8, 1e8]))
            
            point_os = np.array(point_os)
            
            point_os = np.repeat(np.expand_dims(point_os, 
                                                0), 
                                img_size[1],
                                0)
            
            point_os = np.repeat(np.expand_dims(point_os,
                                                0), 
                                img_size[0],
                                0)

            x = np.arange(img_size[0])
            y = np.arange(img_size[1])

            out = np.empty((img_size[0], img_size[1], 2), dtype=int)
            
            out[:, :, 0] = x[:, None]
            out[:, :, 1] = y[None, :]

            
            out = np.repeat(np.expand_dims(out, 2), len(lab), 2)
            sums = np.sqrt(np.sum(np.power(out - point_os, 2), 3))
            sums = np.amax(np.exp(-(sums / self.delta_s)), 2)
            sums = np.expand_dims(sums, 2)
            
            if point == 0:
                s_lab = sums
            else:
                s_lab = np.concatenate([s_lab, sums], 2)
                
        return s_lab

    def generate_group(self, f):
        img, size, scale = load_image(self.image_path+f[0],
                         self.input_shape)
        
        lbs = self.compute_target(img, 
                                  f[1], 
                                  size,
                                  scale)
        
        lbs = pad_img(lbs, self.input_target_shape)
        
        s = lbs.copy()
        s = self.compute_sample_area(s)
        
        s = pad_img(s, self.input_target_shape)
        
        return img, lbs, s

    def next_(self, batch):

        self.img_name = []
        imgs = []
        s_labs = []
        s_lab_coms = []

        if self.generate_type == 'train':
            for i in range(batch):

                if len(self.index) <= batch*2:

                    self.index = list(range(len(self.imgs)))
                    shuffle(self.index)
                    self.epoch += 1

                img, \
                s_lab, \
                s_lab_com\
                    = self.generate_group([self.imgs[self.index[0]],
                                           self.labs[self.index[0]]])

                imgs.append(img)
                s_labs.append(s_lab), \
                s_lab_coms.append(s_lab_com)

                self.index.remove(self.index[0])

        elif self.generate_type == 'test':
            
            if len(self.index) == 0 :
                return []

            img_name = os.path.join(self.image_path +
                                              self.index[0])
            self.img_name.append(img_name)

            img = load_image(img_name,
                             self.input_shape)[0]


            self.index.remove(self.index[0])

            imgs.append(img)
        
        if self.generate_type == 'train':
            return imgs, np.nan_to_num(s_labs), np.nan_to_num(s_lab_coms)

        elif self.generate_type == 'test':
            return imgs

if __name__ == '__main__':
    
    from draw import draw_graph
    generator = generator(input_shape = [368, 368], 
                      sample_rate=0.8, 
                      score_threshold = 0.1)
    
    for i in range(1):
        imgs, \
        labs, \
        com_labs = generator.next_(1)
        imgp = imgs[0].shape
        #if imgp != (368, 368, 3):
        #    input(imgp)
        labs[labs<0.3]=0
        for i in range(labs.shape[3]):
            #pass
            draw_graph(imgs[0], labs[0, :, :, i]).save('test_out_img\\'+str(i)+'.jpg')
