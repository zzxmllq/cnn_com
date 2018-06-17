import h5py
import tensorflow as tf
import numpy as np

pathRoot = '/media/zzxmllq/0002B01D000ED14F/tf/'
filename = "/home/zzxmllq/h5_shapenet_dim32_sdf/h5_shapenet_dim32_sdf/"
writer = tf.python_io.TFRecordWriter(pathRoot + "test_shape_voxel_data.tfrecords")

for i in range(4):
    path = filename + "test_shape_voxel_data" + str(i) + ".h5"
    file = h5py.File(path, 'r')
    data_array = file['data'][:]
    target_array = file['target'][:]
    data_array_v = np.squeeze(data_array[:, 0])[:, :, :, :, np.newaxis]
    data_array_l = np.squeeze(data_array[:, 1])[:, :, :, :, np.newaxis]
    data_array = np.concatenate([data_array_v, data_array_l], axis=4)
    data_array = data_array.astype(np.float32)
    target_array = target_array.reshape((-1, 32, 32, 32, 1))
    target_array = target_array.astype(np.float32)
    for j in range(data_array.shape[0]):
        print(str(i)+':'+str(j))
        data_array_raw = data_array[j].tostring()
        target_array_raw = target_array[j].tostring()
        # print(target_array_raw)
        example = tf.train.Example(
            features=tf.train.Features(
                feature={'data': tf.train.Feature(bytes_list=tf.train.BytesList(value=[data_array_raw])),
                         'target': tf.train.Feature(bytes_list=tf.train.BytesList(value=[target_array_raw]))
                         }
            )
        )
        serialized = example.SerializeToString()
        writer.write(serialized)


writer.close()