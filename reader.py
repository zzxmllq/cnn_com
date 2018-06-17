import tensorflow as tf





def read_and_decode(filename_queue, shuffle_batch=True):
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(
        serialized_example,
        features={
            'data': tf.FixedLenFeature([], tf.string),
            'target': tf.FixedLenFeature([], tf.string)
        }
    )

    data = tf.decode_raw(features['data'], tf.float32)
    data=tf.reshape(data,[32,32,32,2])
    target = tf.decode_raw(features['target'], tf.float32)
    target=tf.reshape(target,[32,32,32,1])
    data_sdf=tf.clip_by_value(tf.abs(data[:,:,:,0]),0,3)
    data_sdf=tf.reshape(data_sdf,[32,32,32,1])
    data_known=tf.reshape(data[:,:,:,1],[32,32,32,1])
    data=tf.concat([data_sdf,data_known],3)
    target=tf.clip_by_value(target,0,3)
    target=tf.log(tf.add(target,tf.constant(1,tf.float32)))

    if shuffle_batch:
        datas, targets = tf.train.shuffle_batch([data, target], batch_size=64, capacity=6400,num_threads=64,
                                                min_after_dequeue=256)
    else:
        datas, targets = tf.train.batch([data, target], batch_size=1, capacity=8000, num_threads=16)
    return datas,targets
