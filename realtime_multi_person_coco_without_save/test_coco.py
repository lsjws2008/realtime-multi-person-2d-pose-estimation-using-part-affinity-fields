#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import tensorflow as tf
# from models import paxel_net
from models import pre_train_vgg
import time
from PIL import Image, ImageDraw
import numpy as np
from math import ceil
import os
from tools.generator_data import generator
from tools.paf_nms import nms
from tools.draw import draw_keypoint, draw_graph, draw_lines
from argparse import ArgumentParser
from tools.connection import connection_of_twoPoints

def initialize_uninitialized(sess):
    global_vars          = tf.global_variables()
    is_not_initialized   = sess.run([tf.is_variable_initialized(var) for var in global_vars])
    not_initialized_vars = [v for (v, f) in zip(global_vars, is_not_initialized) if not f]
    #print(not_initialized_vars)
    if len(not_initialized_vars):
        sess.run(tf.variables_initializer(not_initialized_vars))

parser = ArgumentParser()
parser.add_argument("-s", "--save-log", help="save to train_log", dest="save_log", default="8")
parser.add_argument("-G", "--gpu-memory", help="gpu memary used", type=float, dest="gpu_memory", default="0.1")
args = parser.parse_args()

save_log = os.path.join('D:\\realtime_multi_person_coco\\train_log', 
                        args.save_log)
save_folder = os.path.join(save_log, 'models')

gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=args.gpu_memory)

with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
    net = pre_train_vgg.net(15)

    input_img = tf.placeholder(tf.float32,
                               [None, None, None, 3])
    feature_map = net.vgg_pr_10(input_img, 'train')

    model_pr = tf.train.Saver()
    model_pr.restore(sess, os.path.join(save_folder, 
                                       str(158000) + 'save_net.ckpt'))

    generator = generator(image_path="test_img/",
                          generate_type='test')

    initialize_uninitialized(sess)

    while True:
        begin_time = time.time()

        img = generator.next_(1)
        if len(img) == 0:
            break
        tic = time.time()
        confidence_map = sess.run(feature_map, 
                                  feed_dict={input_img: img})
        
        print('time2:', time.time()-tic)
        confidence = confidence_map[0]
        print(np.amax(confidence))
        points = nms(confidence, 
                     score_threshold=0.6, 
                     score_rate_threshold=0.8)
        """
        graph = np.load('D:/realtime_multi_person_coco/realtime_multi_person_coco_lines/test_out_img/result_'+
                        generator.img_name[0].split('/')[-1][:-4]+'.npy')
        
        lines = connection_of_twoPoints(points, graph, miu=10)
        
        source_img = np.transpose(img[0],
                                  [1, 0, 2]) \
                     * 255
        
        source_img = Image.fromarray(np.uint8(source_img))

        lines = [(np.array(i)*4).tolist() for i in lines]
        
        draw_lines(source_img, lines)
        """
        
        points = [j * 8 for j in points]
        source_img = np.transpose(img[0],
                                  [1, 0, 2]) \
                     * 255
        
        source_img = Image.fromarray(np.uint8(source_img))
        draw_keypoint(source_img, points)
        
        source_img.save('test_out_img/result_' + \
                        generator.img_name[0].split('/')[-1])

        end_time = time.time()
        print('spend time: %.3f,'%(end_time-begin_time), generator.img_name[0])

