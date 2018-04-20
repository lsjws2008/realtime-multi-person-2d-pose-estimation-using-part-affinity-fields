#best model
import test
#import conv_net
import conv_net_bn
import tensorflow as tf
from random import shuffle

imgs, labels, test_imgs, test_labels = test.return_all_data()
batch = 64
ind = list(range(len(imgs)))
shuffle(ind)

def next_(batch):
    global ind
    global labels
    global imgs
    if len(ind) < batch:
        ind = list(range(len(imgs)))
        shuffle(ind)

    img = []
    label = []
    for i in range(batch):
        ins = ind[0]
        img.append(imgs[ins]/255)
        label.append(labels[ins])
        ind.remove(ins)
    return img, label

x_im = tf.placeholder(tf.float32, [None, 32, 32, 3])
x_la = tf.placeholder(tf.float32, [None, 10])
p_train = tf.placeholder(tf.bool, [1])

def compute_accuracy(v_xs, v_ys):
    global prediction
    y_pre = sess.run(prediction, feed_dict={x_im: v_xs, p_train:[False]})
    correct_prediction = tf.equal(tf.argmax(y_pre,1), tf.argmax(x_la,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    result = sess.run(accuracy, feed_dict={x_im: v_xs, x_la: v_ys, p_train:[False]})
    return result

with tf.Session() as sess:
    #net = conv_net.net()
    net = conv_net_bn.net()
    net_out = net.vgg(x_im, p_train)
    ## fc2 layer ##
    prediction = tf.nn.softmax(net_out)
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = x_la, logits = net_out))       # loss
    train_step = tf.train.AdamOptimizer(0.001).minimize(cross_entropy)
    
    merged = tf.summary.merge_all()
    writer = tf.summary.FileWriter('logs/',sess.graph)
    saver = tf.train.Saver()

    init = tf.global_variables_initializer()
    sess.run(init)
    for i in range(100000):
        batch_xs, batch_ys = next_(batch)
        #print(sess.run(cross_entropy, feed_dict={x_im: batch_xs, x_la: batch_ys, p_train:[True]}))
        sess.run(train_step, feed_dict={x_im: batch_xs, x_la: batch_ys, p_train:[True]})
        if i % 50 == 0:
            acc = compute_accuracy(test_imgs[:100], test_labels[:100])
            cs = sess.run(cross_entropy, feed_dict={x_im: batch_xs, x_la: batch_ys, p_train:[False]})
            print(i, cs, acc)
        if i > 20000 and acc >= 0.9:
            break
    saver.save(sess, 'save/save_net.ckpt')