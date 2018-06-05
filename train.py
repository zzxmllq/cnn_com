import tensorflow as tf
import tensorflow.contrib.slim as slim
import reader
import losses
import epnbase

flags = tf.flags
FLAGS = flags.FLAGS
flags.DEFINE_integer('origin_col', 32, 'origin column of data/target')
flags.DEFINE_integer('origin_row', 32, 'origin row of data/target')
flags.DEFINE_float('learning_rate', 0.0001, 'learning rate')
flags.DEFINE_float('min_learning_rate', 0.00001, 'Min learning rate.')
flags.DEFINE_string('data_path', '/home/zzxmllq/h5_shapenet_dim32_sdf/h5_shapenet_dim32_sdf/',
                    'Training data file path.')
flags.DEFINE_integer('batch_size', 64, 'Number of blocks in each batch')
flags.DEFINE_integer('max_epoch', 150, 'epochs to run')
root_path = '/media/zzxmllq/0002B01D000ED14F/tf/'
tfrecord_filename = root_path + 'train_shape_voxel_data.tfrecords'
model_path="/home/zzxmllq/h5_shapenet_dim32_sdf/h5_shapenet_dim32_sdf/model/521model.ckpt"

def train():
    filename_queue = tf.train.string_input_producer([tfrecord_filename])
    data_array, target_array = reader.read_and_decode(filename_queue)
    with tf.device('/gpu:0'):
        global_step = slim.create_global_step()
    with tf.device('/cpu:0'):
        lrn_rate = tf.train.exponential_decay(
            FLAGS.learning_rate, global_step, 5000, 0.917, staircase=True)
        tf.summary.scalar('learning_rate', lrn_rate)

    with tf.device('/gpu:0'):
        optimizer = tf.train.AdamOptimizer(lrn_rate)
        # data = tf.placeholder(tf.float32, [FLAGS.batch_size, 32, 32, 32, 2])
        # target = tf.placeholder(tf.float32, [FLAGS.batch_size, 32, 32, 32, 1])
        ops = epnbase.model(input_data=data_array)
        loss, mask = losses.get_l1_loss(input_data=data_array, pre_data=ops, label=target_array)
        #tf.summary.scalar('loss_function', loss)
        train_step = optimizer.minimize(loss=loss, global_step=global_step)
    #with tf.device('/cpu:0'):
        #tf.summary.scalar('loss_function', loss)
        #saver = tf.train.Saver()

    # print target_array[0][0][0][0][0]
    # sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=True))

    # print loss.get_shape().as_list()
    with tf.Session() as sess:
        sess.run(tf.local_variables_initializer())
        sess.run(tf.global_variables_initializer())
        #merged_summary_op = tf.summary.merge_all()
        #summary_writer = tf.summary.FileWriter('/home/zzxmllq/h5_shapenet_dim32_sdf/h5_shapenet_dim32_sdf', sess.graph)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        # for j in range(FLAGS.max_epoch):
        for j in range(FLAGS.max_epoch):
            for i in range(2000):
                _, loss_value, mask_v = sess.run([train_step, loss, mask])
                print str(j + 1) + ' epoch' + ' ' + str(i) + ' minibatch' + ':' + str(loss_value)
                #if i % 10 == 0:
                    #summary_str=sess.run(merged_summary_op)
                    #summary_writer.add_summary(summary_str,j*2000+i)
        #save_path=saver.save(sess,model_path)
        #print("Model saved in file:%s"% save_path)
        coord.request_stop()
        coord.join(threads)



def main(_):
    train()


if __name__ == '__main__':
    tf.app.run()
