import tensorflow as tf
import paxel_net
import pre_train_vgg
import time
#from math import pow
import os
from random import shuffle
import numpy as np
from PIL import Image

def initialize_uninitialized(sess):
    global_vars          = tf.global_variables()
    is_not_initialized   = sess.run([tf.is_variable_initialized(var) for var in global_vars])
    not_initialized_vars = [v for (v, f) in zip(global_vars, is_not_initialized) if not f]
    #print(not_initialized_vars)
    if len(not_initialized_vars):
        sess.run(tf.variables_initializer(not_initialized_vars))

batch = 10
epoch = 100000

files = []
root_path = "F:/trains/"

def next_(batch):
    global files
    imgs = []
    s_labs = []
    s_lab_coms = []
    l_labs = []
    l_lab_coms = []
    for i in range(batch):
        if len(files) == 0:
            files = os.listdir(root_path)
            shuffle(files)
        f = os.listdir(root_path+files[0]+'/')
        for fs in f:
            path = root_path+files[0]+'/'+fs
            if fs[:3] == 'img':
                img = np.array(Image.open(path).resize((800,400)))/255
                #print(img.shape, len(img.shape))
                if len(img.shape) == 2:
                    img = np.expand_dims(img, 2)
                    img = np.concatenate((img, img, img), 2)
                imgs.append(np.transpose(img, (1, 0, 2)))
            elif fs[:3] == 's_l':
                s = np.load(path)
                s_labs.append(s)
                for keypoint in range(17):
                    sums = s[:, :, keypoint]
                    count = np.count_nonzero(sums)
                    com_lab = np.copy(sums)
                    com_lab[com_lab>=0.1]=1
                    x = list(range(100))
                    y = list(range(50))
                    while(count>0):
                        loc_x = np.random.choice(x,1)
                        loc_y = np.random.choice(y,1)
                        if com_lab[loc_x, loc_y] == 0:
                            com_lab[loc_x, loc_y] = 1
                            count-=1
                    if keypoint == 0:
                        s_com_lab = np.expand_dims(com_lab, 2)
                    else:
                        s_com_lab = np.concatenate((s_com_lab, np.expand_dims(com_lab, 2)), 2)
                s_lab_coms.append(s_com_lab)
            elif fs[:3] == 'l_l':
                s = np.load(path)
                l_labs.append(s+1)
                for line in range(24):
                    sums = s[:, :, line]
                    count = np.count_nonzero(sums)
                    com_lab = np.copy(sums)
                    com_lab[com_lab>=0.1]=1
                    x = list(range(100))
                    y = list(range(50))
                    begin_time = time.time()
                    while(count>0):
                        loc_x = np.random.choice(x,1)
                        loc_y = np.random.choice(y,1)
                        if com_lab[loc_x, loc_y] == 0:
                            com_lab[loc_x, loc_y] = 1
                            count-=1
                        end_time = time.time()
                        if end_time - begin_time > 2:
                            print("out")
                            break
                    if line == 0:
                        l_com_lab = np.expand_dims(com_lab, 2)
                    else:
                        l_com_lab = np.concatenate((l_com_lab, np.expand_dims(com_lab, 2)), 2)
                l_lab_coms.append(l_com_lab)
        files.remove(files[0])
    return imgs, s_labs, s_lab_coms, l_labs, l_lab_coms

with tf.Session() as sess:
    net = pre_train_vgg.net()
    #model_pr = tf.train.Saver()
    #model_pr.restore(sess, 'save/save_net.ckpt')
    paxel = paxel_net.paxel()
    img = tf.placeholder(tf.float32, [None, 800, 400, 3])
    s_lab = tf.placeholder(tf.float32, [None, 100, 50, 17])
    s_com_lab = tf.placeholder(tf.float32, [None, 100, 50, 17])
    l_lab = tf.placeholder(tf.float32, [None, 100, 50, 24])
    l_com_lab = tf.placeholder(tf.float32, [None, 100, 50, 24])
    lr = tf.placeholder(tf.float32, [None])
    feature_map = net.vgg_pr_10(img, 'train')
    output_s, output_l = paxel.S_net(feature_map, 'train')
    model_af = tf.train.Saver()
    lo_s = (output_s-s_lab)*s_com_lab
    lo_l = (output_l-l_lab)*l_com_lab
    s_lo_s = tf.reduce_sum(tf.abs(lo_s))
    s_lo_l = tf.reduce_sum(tf.abs(lo_l))
    loss_s = tf.sqrt(tf.reduce_sum(tf.pow(lo_s, 2)))
    loss_l = tf.sqrt(tf.reduce_sum(tf.pow(lo_l, 2)))
    loss = loss_s + loss_l
    train_step = tf.train.AdamOptimizer(0.001).minimize(loss)
    initialize_uninitialized(sess)
    #learning_rate = 0.1
    #lr_decay = 0.1
    print('begin')
    for seq in range(epoch):
        lear = 0.001
        #learning_rate = learning_rate * pow(1+lr_decay, seq)
        print('\nseq:',str(seq))
        
        if seq == 20000:
            lear = 0.00001
        
        if seq == 3000:
            lear = 0.0001
        begin_time = time.time()
        imgs, s_labs, s_com_labs, l_labs, l_com_labs = next_(batch)
        sess.run(train_step, feed_dict = {img:imgs, s_lab:s_labs, s_com_lab: s_com_labs
                                          , l_lab:l_labs, l_com_lab: l_com_labs,lr: [lear]})
        end_time = time.time()
        print('time ',str(end_time-begin_time))
        if seq%5 == 0:
                print('loss_s = ', "%5.3f"%(sess.run(s_lo_s, feed_dict = {img:imgs, s_lab:s_labs, s_com_lab: s_com_labs
                                                             , l_lab:l_labs, l_com_lab: l_com_labs})), 
                    ' s_total = ', "%5.3f"%(np.sum(np.array(s_labs))), 
                    ' loss_l = ', "%5.3f"%(sess.run(s_lo_l, feed_dict = {img:imgs, s_lab:s_labs, s_com_lab: s_com_labs
                                                             , l_lab:l_labs, l_com_lab: l_com_labs})), 
                    ' l_total = ', "%5.3f"%(np.sum(np.absolute(np.array(l_labs))))
                    )
        if seq%100 == 0:
            model_af.save(sess, 'save_all/save_net.ckpt')
