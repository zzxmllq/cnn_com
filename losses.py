import tensorflow as tf


# def get_l1_loss(input_data, pre_data, label):
#     masks = tf.cast(tf.equal(input_data[:, :, :, :, 1], -1), tf.float32)
#
#
#     masks = tf.reshape(masks, [64, 32, 32, 32, 1])
#     print masks.get_shape().as_list()
#     num_1 = tf.reduce_sum(masks)
#     print num_1.get_shape().as_list()
#     pre_data_loss = tf.multiply(masks, pre_data)
#     label_loss = tf.multiply(masks, label)
#     # sum_l1_loss=tf.reduce_sum(tf.abs(pre_data_loss-label_loss))
#     # l1_loss=tf.divide(sum_l1_loss,num_1)
#     l1_loss = tf.abs(pre_data_loss - label_loss)
#     sign_1 = tf.constant(1, tf.float32)
#     mask_1 = tf.cast(tf.less(l1_loss, sign_1), tf.float32)
#     sum_smooth_l1_loss_1 = tf.reduce_sum(tf.divide(tf.square(tf.multiply(mask_1, l1_loss)), tf.constant(2, tf.float32)))
#     sign_0 = tf.constant(0, tf.float32)
#     mask_2 = tf.cast(tf.equal(mask_1, sign_0), tf.float32)
#     sum_smooth_l1_loss_2 = tf.reduce_sum(tf.multiply(mask_2, l1_loss) - tf.constant(0.5, tf.float32))
#     sum_smooth_l1_loss = tf.divide(sum_smooth_l1_loss_1 + sum_smooth_l1_loss_2, num_1)
#
#     return sum_smooth_l1_loss, masks
def get_l1_loss(input_data, pre_data, label):
    masks = tf.cast(tf.equal(input_data[:, :, :, :, 1], -1), tf.float32)
    masks = tf.reshape(masks, [-1, 32, 32, 32, 1])
    l1_loss = tf.losses.absolute_difference(labels=label, predictions=pre_data, weights=masks)

    return l1_loss, masks
