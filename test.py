import tensorflow as tf
import tensorflow.contrib.slim as slim
import reader
import losses
import epn_test
import os
import numpy as np
import scipy.io as sio

flags = tf.flags
FLAGS = flags.FLAGS
flags.DEFINE_integer('origin_col', 32, 'origin column of data/target')
flags.DEFINE_integer('origin_row', 32, 'origin row of data/target')
flags.DEFINE_float('learning_rate', 0.001, 'learning rate')
flags.DEFINE_float('min_learning_rate', 0.0001, 'Min learning rate.')
flags.DEFINE_string('data_path', '/home/zzxmllq/h5_shapenet_dim32_sdf/h5_shapenet_dim32_sdf/',
                    'Training data file path.')
flags.DEFINE_integer('batch_size', 1, 'Number of blocks in each batch')
flags.DEFINE_integer('max_epoch', 1, 'epochs to run')
flags.DEFINE_string('output_path', '/media/zzxmllq/0002B01D000ED14F/tf',
                    'output file path.')
flags.DEFINE_integer('pad_test', 0, 'Scene padding.')
model_path = "/home/zzxmllq/h5_shapenet_dim32_sdf/h5_shapenet_dim32_sdf/model/epn_model.ckpt"
root_path = '/media/zzxmllq/0002B01D000ED14F/tf/'
tfrecord_filename = root_path + 'test_shape_voxel_data.tfrecords'


def save_mat_df(df, error, filename):
    """Saves df as matlab .mat file."""
    output = {'x': df}
    if error is not None:
        output['errors'] = error
    sio.savemat(filename, output)


def save_iso_meshes(dfs, errs, semantics, filenames, isoval=1):
    """Saves dfs to obj files (by calling matlab's 'isosurface' function)."""
    assert len(dfs) == len(filenames) and (
            errs is None or len(dfs) == len(errs)) and (semantics is None or
                                                        len(dfs) == len(semantics))
    # Save semantics meshes if applicable.

    mat_filenames = [os.path.splitext(x)[0] + '.mat' for x in filenames]
    # Save .mat files for matlab call.
    command = ""
    for i in range(len(filenames)):
        if dfs[i] is None:
            continue
        err = None if errs is None else errs[i]
        save_mat_df(dfs[i], err, mat_filenames[i])
        command += "mat_to_obj('{0}', '{1}', {2});".format(mat_filenames[i],
                                                           filenames[i], isoval)
    command += 'exit;'

    tf.logging.info(
        'matlab -nodisplay -nosplash -nodesktop -r "{0}"'.format(command))
    # Execute matlab.
    os.system('matlab -nodisplay -nosplash -nodesktop -r "{0}"'.format(command))
    # os.system('matlab -r "{0}"'.format(command))
    # Clean up .mat files.
    for i in range(len(mat_filenames)):
        os.system('rm -f {0}'.format(mat_filenames[i]))


def export_prediction_to_mesh(outprefix, input_sdf, output_df,
                              target_df):
    (scene_dim_z, scene_dim_y, scene_dim_x) = input_sdf.shape
    save_input_sdf = 3 * np.ones(
        [scene_dim_z, scene_dim_y, scene_dim_x])
    save_prediction = np.copy(save_input_sdf)
    save_target = None if target_df is None else np.copy(save_input_sdf)
    save_input_sdf[:, :, :] = input_sdf
    save_prediction[:, :, :] = output_df
    if target_df is not None:
        save_target[:, :, :] = target_df
        # For error visualization as colors on mesh.
        save_errors = np.zeros(shape=save_prediction.shape)
        save_errors[:, :, :] = np.abs(
            output_df - target_df)
    save_iso_meshes(
        [save_input_sdf, save_prediction, save_target],
        [None, save_errors, save_errors], None,
        [
            outprefix + 'input.obj', outprefix + 'pred.obj',
            outprefix + 'target.obj'
        ],
        isoval=1)


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
        ops = epn_test.model(input_data=data_array)
        loss, mask = losses.get_l1_loss(input_data=data_array, pre_data=ops, label=target_array)
        # tf.summary.scalar('loss_function', loss)
        # train_step = optimizer.minimize(loss=loss, global_step=global_step)
    with tf.device('/cpu:0'):
        tf.summary.scalar('loss_function', loss)
        saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver.restore(sess, model_path)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        loss_value_total = 0.0
        for i in range(1):
            loss_value_total = loss_value_total + sess.run(loss)
            data_origin,pre_origin,target_origin = sess.run([data_array,ops,target_array])
            # loss_value_total=tf.concat([loss_value_total,loss_value],axis=0)
            print str(i) + 'step'
            # print type(loss_value)
        print 'loss_mean=' + str(loss_value_total)
        # input = tf.reshape(data_array[:, :, :, :, 0], [32, 32, 32])
        # pre = tf.subtract(tf.exp(pre), tf.constant(1, tf.float32))
        # pre = tf.reshape(pre, [32, 32, 32])
        # target = tf.subtract(tf.exp(target_array), tf.constant(1, tf.float32))
        # target = tf.reshape(target, [32, 32, 32])
        input=np.reshape(data_origin[:,:,:,:,0],(32,32,32))
        pre=np.exp(pre_origin)-1
        pre=np.reshape(pre,[32,32,32])
        target=np.exp(target_origin)-1
        target=np.reshape(target,[32,32,32])
        outprefix = os.path.join(
            FLAGS.output_path, 'sample_')
        export_prediction_to_mesh(outprefix, input, pre, target)

        coord.request_stop()
        coord.join(threads)


def main(_):
    test()


if __name__ == '__main__':
    tf.app.run()
