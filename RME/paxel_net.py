import tensorflow as tf
from pre_train_vgg import weight_variable, bias_variable, conv2d, batch_norm

class paxel():
    def __init__(self):
        self.W_conv1 = weight_variable('w_conv_1' ,[3, 3, 256, 128]) # patch 5x5, in size 1, out size 32
        self.b_conv1 = bias_variable('b_conv_1',[128])
        self.W_conv2 = weight_variable('w_conv_2',[3, 3, 128, 128]) # patch 5x5, in size 1, out size 32
        self.b_conv2 = bias_variable('b_conv_2',[128])
        self.W_conv3 = weight_variable('w_conv_3',[3, 3, 128, 128]) # patch 5x5, in size 1, out size 32
        self.b_conv3 = bias_variable('b_conv_3',[128])
        self.W_conv4 = weight_variable('w_conv_4',[1, 1, 128, 512]) # patch 5x5, in size 1, out size 32
        self.b_conv4 = bias_variable('b_conv_4',[512])
        self.W_conv5 = weight_variable('w_conv_5',[1, 1, 512, 17]) # patch 5x5, in size 1, out size 32
        self.b_conv5 = bias_variable('b_conv_5',[17])
        #self.W_conv6 = weight_variable('w_conv_5',[1, 1, 256, 17]) # patch 5x5, in size 1, out size 32
        #self.b_conv6 = bias_variable('b_conv_5',[17])
    
        self.lW_conv1 = weight_variable('lw_conv_1',[3, 3, 256, 128]) # patch 5x5, in size 1, out size 32
        self.lb_conv1 = bias_variable('lb_conv_1',[128])
        self.lW_conv2 = weight_variable('lw_conv_2',[3, 3, 128, 128]) # patch 5x5, in size 1, out size 32
        self.lb_conv2 = bias_variable('lb_conv_2',[128])
        self.lW_conv3 = weight_variable('lw_conv_3',[3, 3, 128, 128]) # patch 5x5, in size 1, out size 32
        self.lb_conv3 = bias_variable('lb_conv_3',[128])
        self.lW_conv4 = weight_variable('lw_conv_4',[1, 1, 128, 512]) # patch 5x5, in size 1, out size 32
        self.lb_conv4 = bias_variable('lb_conv_4',[512])
        self.lW_conv5 = weight_variable('lw_conv_5',[1, 1, 512, 24]) # patch 5x5, in size 1, out size 32
        self.lb_conv5 = bias_variable('lb_conv_5',[24])
        #self.lW_conv6 = weight_variable('lw_conv_5',[1, 1, 256, 24]) # patch 5x5, in size 1, out size 32
        #self.lb_conv6 = bias_variable('lb_conv_5',[24])
        
    def S_net(self, feature_map, phase_train):
        h_conv1 = conv2d(feature_map, self.W_conv1) + self.b_conv1 # output size 28x28x32
        h_conv1_bn = tf.nn.relu(batch_norm(h_conv1, 128, phase_train))
        h_conv2 = conv2d(h_conv1_bn, self.W_conv2) + self.b_conv2 # output size 28x28x32
        h_conv2_bn = tf.nn.relu(batch_norm(h_conv2, 128, phase_train))
        h_conv3 = conv2d(h_conv2_bn, self.W_conv3) + self.b_conv3 # output size 28x28x32
        h_conv3_bn = tf.nn.relu(batch_norm(h_conv3, 128, phase_train))
        h_conv4 = conv2d(h_conv3_bn, self.W_conv4) + self.b_conv4 # output size 28x28x32
        h_conv4_bn = tf.nn.relu(batch_norm(h_conv4, 512, phase_train))
        h_conv5 = conv2d(h_conv4_bn, self.W_conv5) + self.b_conv5 # output size 28x28x32
        h_conv5_bn = tf.nn.relu(batch_norm(h_conv5, 17, phase_train))
        #h_conv6 = conv2d(h_conv5_bn, self.W_conv6) + self.b_conv6 # output size 28x28x32
        #h_conv6_bn = tf.nn.relu(batch_norm(h_conv6, 17, phase_train))
        
        lh_conv1 = conv2d(feature_map, self.lW_conv1) + self.lb_conv1 # output size 28x28x32
        lh_conv1_bn = tf.nn.relu(batch_norm(lh_conv1, 128, phase_train))
        lh_conv2 = conv2d(lh_conv1_bn, self.lW_conv2) + self.lb_conv2 # output size 28x28x32
        lh_conv2_bn = tf.nn.relu(batch_norm(lh_conv2, 128, phase_train))
        h_conv3 = conv2d(h_conv2_bn, self.W_conv3) + self.b_conv3 # output size 28x28x32
        h_conv3_bn = tf.nn.relu(batch_norm(h_conv3, 128, phase_train))
        lh_conv4 = conv2d(lh_conv2_bn, self.lW_conv4) + self.lb_conv4 # output size 28x28x32
        lh_conv4_bn = tf.nn.relu(batch_norm(lh_conv4, 512, phase_train))
        lh_conv5 = conv2d(lh_conv4_bn, self.lW_conv5) + self.lb_conv5 # output size 28x28x32
        lh_conv5_bn = tf.nn.relu(batch_norm(lh_conv5, 24, phase_train))
        #lh_conv6 = conv2d(lh_conv5_bn, self.lW_conv6) + self.lb_conv6 # output size 28x28x32
        #lh_conv6_bn = tf.nn.relu(batch_norm(lh_conv6, 24, phase_train))
        
        return h_conv5_bn, lh_conv5_bn
        