from argparse import ArgumentParser
import tensorflow as tf
from models import pre_train_vgg
import time
#from math import pow
import os
from random import shuffle
import numpy as np
from PIL import Image
import math
import matplotlib.pyplot as plt
from tools.generator_data import generator

def initialize_uninitialized(sess):
    global_vars          = tf.global_variables()
    is_not_initialized   = sess.run([tf.is_variable_initialized(var) for var in global_vars])
    not_initialized_vars = [v for (v, f) in zip(global_vars, is_not_initialized) if not f]
    print(not_initialized_vars)
    if len(not_initialized_vars):
        sess.run(tf.variables_initializer(not_initialized_vars))

batch = 12
epoch = 1000000
lear = 4e-5
lss = []
generator = generator(input_shape = [368, 368],
                      input_target_shape=(46, 46),
                      sample_rate=1,
                      score_threshold = 0.3, 
                      delta_s = 3)

parser = ArgumentParser()
parser.add_argument("-s", "--save-log", help="save to train_log", dest="save_log", default="9")
parser.add_argument("-G", "--gpu-memory", help="gpu memary used", type=float, dest="gpu_memory", default="0.4")
args = parser.parse_args()

save_log = os.path.join('D:\\realtime_multi_person_coco\\train_log',
                        args.save_log)

if not os.path.isdir(save_log):
    os.mkdir(save_log)

gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=args.gpu_memory)

output_num = 15

with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:

    net = pre_train_vgg.net(output_num)

    img = tf.placeholder(tf.float32,
                         [batch, None, None, 3])
    lab = tf.placeholder(tf.float32,
                         [batch, None, None, output_num])
    com_lab = tf.placeholder(tf.float32,
                             [batch, None, None, output_num])
    lr = tf.placeholder(tf.float32,
                        [None])

    output = net.vgg_pr_10(img, 'train')

    lo = (output-lab)*com_lab
    loss = tf.reduce_sum(tf.abs(lo))
    ls_p = loss / tf.reduce_sum(com_lab)

    train_step = tf.train.MomentumOptimizer(lr[0],
                                            0.9).\
        minimize(loss)

    #model_pr = tf.train.Saver()
    
    #model_pr.restore(sess, 'save_all/save_net.ckpt')

    model_af = tf.train.Saver()
    initialize_uninitialized(sess)

    print('begin', end=' :')
    for seq in range(epoch):

        if (seq+1)%20000 == 0:
            lear *= 0.5
            print(lear)

        begin_time = time.time()

        imgs, \
        labs, \
        com_labs = generator.next_(batch)

        sess.run(train_step,
                 feed_dict = {img:imgs,
                              lab:labs,
                              com_lab: com_labs,
                              lr: [lear]})

        if seq%5 == 0:

            print('\nSequence:',str(seq))

            [ls_t,
             out] = sess.run([ls_p,
                              output],
                             feed_dict={img: imgs,
                                        lab: labs,
                                        com_lab: com_labs,
                                        lr: [lear]})

            lss.append(ls_t)
            if len(lss) > 1e4:
                lss.remove(lss[0])

            plt.plot(range(len(lss)), lss)
            feed_back_folder = os.path.join(save_log, 'feed_back')

            if not os.path.isdir(feed_back_folder):
                os.mkdir(feed_back_folder)
            plt.savefig(os.path.join(feed_back_folder,
                                     'l'+str(int(seq/5e4))+'.png'))
            plt.show()
            plt.clf()

            if np.isnan(ls_t):
                input('isnan')

            avg_loss = sum([0.9*\
                            math.pow(0.1,
                                     len(lss)-1-lsins)\
                            * ls \
                            for lsins, ls in enumerate(lss)])

            print('spand time: {0:.3f}, loss: {1:.3f}, max value: {2:.3f}'.\
                  format(time.time() - begin_time
                         , avg_loss, 
                         np.amax(out)
                         ))

        if (seq+1)%1000 == 0:
            save_folder = os.path.join(save_log, 'models')

            if not os.path.isdir(save_folder):
                os.mkdir(save_folder)

            model_af.save(sess,
                          os.path.join(save_folder,
                                       str(seq+1) + 'save_net.ckpt'))
