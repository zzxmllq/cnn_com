import tensorflow as tf
import tensorflow.contrib.slim as slim
import reader
import losses
import epnbase

flags = tf.flags
FLAGS = flags.FLAGS
flags.DEFINE_integer('origin_col', 32, 'origin column of data/target')
flags.DEFINE_integer('origin_row', 32, 'origin row of data/target')
flags.DEFINE_float('learning_rate', 0.001, 'learning rate')
flags.DEFINE_float('min_learning_rate', 0.0001, 'Min learning rate.')
flags.DEFINE_string('data_path', '/home/zzxmllq/h5_shapenet_dim32_sdf/h5_shapenet_dim32_sdf/',
                    'Training data file path.')
flags.DEFINE_integer('batch_size', 64, 'Number of blocks in each batch')
flags.DEFINE_integer('max_epoch', 1, 'epochs to run')
model_path = "/home/zzxmllq/h5_shapenet_dim32_sdf/h5_shapenet_dim32_sdf/model/521model.ckpt"
root_path = '/media/zzxmllq/0002B01D000ED14F/tf/'
tfrecord_filename = root_path + 'test_shape_voxel_data.tfrecords'


def test():
    filename_queue = tf.train.string_input_producer([tfrecord_filename])
    data_array, target_array = reader.read_and_decode(filename_queue, shuffle_batch=False)
    with tf.device('/gpu:0'):
        global_step = slim.create_global_step()
    with tf.device('/cpu:0'):
        lrn_rate = tf.train.exponential_decay(
            FLAGS.learning_rate, global_step, 5000, 0.917, staircase=True)
        tf.summary.scalar('learning_rate', lrn_rate)

    with tf.device('/gpu:0'):
        # tf.train.exponential_decay(
        #     FLAGS.learning_rate, global_step, 5000, 0.92, staircase=True)
        # visualizing learning
        # tf.summary.scalar('learning_rate', lrn_rate)
        optimizer = tf.train.AdamOptimizer(lrn_rate)
        # data = tf.placeholder(tf.float32, [FLAGS.batch_size, 32, 32, 32, 2])
        # target = tf.placeholder(tf.float32, [FLAGS.batch_size, 32, 32, 32, 1])
        ops = epnbase.model(input_data=data_array)
        loss, mask = losses.get_l1_loss(input_data=data_array, pre_data=ops, label=target_array)
        # tf.summary.scalar('loss_function', loss)
        train_step = optimizer.minimize(loss=loss, global_step=global_step)
    with tf.device('/cpu:0'):
        tf.summary.scalar('loss_function', loss)
        saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver.restore(sess, model_path)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        loss_value_total = 0.0
        for i in range(470):
            loss_value_total=loss_value_total+sess.run(loss)
            #loss_value_total=tf.concat([loss_value_total,loss_value],axis=0)
            print str(i) + 'step'
            # print type(loss_value)
        print 'loss_mean='+str(loss_value_total/470)
        coord.request_stop()
        coord.join(threads)


def main(_):
    test()


if __name__ == '__main__':
    tf.app.run()
