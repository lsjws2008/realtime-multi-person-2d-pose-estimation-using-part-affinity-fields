
import tensorflow as tf

def weighte_l2_loss(output, lab, com_lab):

    lo = (output - lab) * com_lab
    lo_p = tf.reduce_sum(tf.abs(lo)) / tf.reduce_sum(com_lab)
    loss = tf.reduce_sum(tf.abs(lo))
    ls_p = loss / tf.reduce_sum(com_lab)
