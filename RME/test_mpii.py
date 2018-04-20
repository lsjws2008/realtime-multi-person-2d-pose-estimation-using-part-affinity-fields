#!/usr/bin/env python2
# -*- coding: utf-8 -*-
import tensorflow as tf
import paxel_net
import pre_train_vgg
import time
from PIL import Image
import numpy as np
from math import ceil

def initialize_uninitialized(sess):
    global_vars          = tf.global_variables()
    is_not_initialized   = sess.run([tf.is_variable_initialized(var) for var in global_vars])
    not_initialized_vars = [v for (v, f) in zip(global_vars, is_not_initialized) if not f]
    print(not_initialized_vars)
    if len(not_initialized_vars):
        sess.run(tf.variables_initializer(not_initialized_vars))


with tf.Session() as sess:
    net = pre_train_vgg.net()
    paxel = paxel_net.paxel()
    img = tf.placeholder(tf.float32, [None, None, None, 3])
    feature_map = net.vgg_pr_10(img, 'train')
    output_s, output_l = paxel.S_net(feature_map, 'train')
    model_pr = tf.train.Saver()
    model_pr.restore(sess, 'save_all/save_net.ckpt')
    for i in range(17):
        image = i
        begin_time = time.time()
        im = Image.open('test_img/'+str(image)+'.jpg')
        im_data = np.array(im).transpose((1, 0, 2))#/255
        #initialize_uninitialized(sess)
        confidence_map = sess.run(output_s, feed_dict = {img:[im_data]})
        end_time = time.time()
        print('spend time: %.3f'%(end_time-begin_time))
        #im.resize((320, 180)).show()
        confidence_map = confidence_map[0]
        maps = []
        for i in range(17):
            map_0 = confidence_map[:, :, i]
            map_0[map_0 < 0.6] = 0
            mas = np.amax(map_0)
            if mas>0:
                map_0 = map_0/mas
            map_0 = np.expand_dims(map_0, 2)
            map_0 = map_0*255
            #ims = np.concatenate((im[:, :, 0:1], im[:,:,1:2], ims), 2)
            maps.append(map_0)
            #im = Image.fromarray(np.uint8(map_0).transpose((1, 0, 2)))
            #im.show()
            #im.save('test_out_img/'+str(i)+'.jpg')
        
        maps = np.sum(maps, 0)
        print(np.amax(maps))
        im = Image.open('test_img/'+str(image)+'.jpg')
        w, h = im.size
        im = im.resize((ceil(w/8), ceil(h/8)))
        im = np.array(im).transpose((1, 0, 2))
        ims = im[:,:,2:] + maps
        ims = np.concatenate((im[:, :, 0:1], im[:,:,1:2], ims), 2)
        ims[:,:,2] = np.clip(ims[:,:,2], 0, 255)
        ims = np.transpose(ims, (1, 0, 2))
        im = Image.fromarray(np.uint8(ims))
        im.save('test_out_img/'+str(image)+'.jpg')
        
        ims = np.concatenate([maps, maps, maps], 2)
        ims = np.transpose(ims, (1, 0, 2))
        ims = np.clip(ims, 0, 255)
        print(ims.shape)
        ims = Image.fromarray(np.uint8(ims))
        ims.save('test_out_img/'+str(image+16)+'.jpg')
        