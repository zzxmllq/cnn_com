import tensorflow as tf
import tensorflow.contrib.slim as slim
import write
import model
import losses

flags = tf.flags
FLAGS = flags.FLAGS
flags.DEFINE_integer('origin_col', 32, 'origin column of data/target')
flags.DEFINE_integer('origin_row', 32, 'origin row of data/target')
flags.DEFINE_float('learning_rate', 0.001, 'learning rate')
flags.DEFINE_float('min_learning_rate', 0.0001, 'Min learning rate.')
flags.DEFINE_string('data_path', '/home/zzxmllq/h5_shapenet_dim32_sdf/h5_shapenet_dim32_sdf/train_shape_voxel_data0.h5',
                    'Training data file path.')
flags.DEFINE_integer('batch_size', 10, 'Number of blocks in each batch')


def train():
    data_array, target_array = write.read(FLAGS.data_path)
    with tf.device('/gpu:0'):
        global_step = slim.create_global_step()
    with tf.device('/cpu:0'):
        lrn_rate = tf.train.exponential_decay(
          FLAGS.learning_rate, global_step, 5000, 0.92, staircase=True)

    with tf.device('/gpu:0'):
            # tf.train.exponential_decay(
            #     FLAGS.learning_rate, global_step, 5000, 0.92, staircase=True)
        # visualizing learning
        #tf.summary.scalar('learning_rate', lrn_rate)
        optimizer = tf.train.AdamOptimizer(lrn_rate)
        data = tf.placeholder(tf.float32, [FLAGS.batch_size, 32, 32, 32, 2])
        target = tf.placeholder(tf.float32, [FLAGS.batch_size, 32, 32, 32, 1])
        ops = model.model(input_data=data)
        loss, mask = losses.get_l1_loss(input_data=data, pre_data=ops, label=target)
        train_step = optimizer.minimize(loss=loss, global_step=global_step)

    # print target_array[0][0][0][0][0]
    # sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=True))

    # print loss.get_shape().as_list()
    with tf.Session() as sess:

        sess.run(tf.global_variables_initializer())
        for j in range(100):
            for i in range(1000):
                _, loss_value, mask_v = sess.run([train_step, loss, mask], feed_dict={
                    data: data_array[i * FLAGS.batch_size:i * FLAGS.batch_size + FLAGS.batch_size],
                    target: target_array[i * FLAGS.batch_size:i * FLAGS.batch_size + FLAGS.batch_size]})
                print loss_value

    # min_after_dequeue=1000
    # batch_size=10
    # capacity=min_after_dequeue+3*batch_size
    # data_batch,target_batch=tf.train.shuffle_batch([data,target],batch_size=batch_size,capacity=capacity,min_after_dequeue=min_after_dequeue)
    # print type(data)
    # ops=model.model(input_data=data,input_target=target)
    # data=tf.convert_to_tensor(data)
    # target=tf.convert_to_tensor(target)
    # print type(target[0][0][0][0][0])


#  inputs_queue = slim.python.slim.data.prefetch_queue.prefetch_queue(
#     (data,target))
# def tower_fn(inputs_queue):
#     data,target=inputs_queue.dequeue()
#    ops = model.model(input_data=data, input_target=target)

def main(_):
    train()


if __name__ == '__main__':
    tf.app.run()
