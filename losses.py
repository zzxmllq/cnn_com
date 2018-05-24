import tensorflow as tf
import tensorflow.contrib.slim as slim

def get_l1_loss(input_data,pre_data,label):
    masks=tf.cast(tf.equal(input_data[:,:,:,:,1],-1),tf.float32)
    masks=tf.reshape(masks,[10,32,32,32,1])
    print masks.get_shape().as_list()
    num_1=tf.reduce_sum(masks)
    print num_1.get_shape().as_list()
    pre_data_loss=tf.multiply(masks,pre_data)
    label_loss=tf.multiply(masks,label)
    sum_l1_loss=tf.reduce_sum(tf.abs(pre_data_loss-label_loss))
    l1_loss=tf.divide(sum_l1_loss,num_1)
    return l1_loss,masks
