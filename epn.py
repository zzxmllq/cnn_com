import tensorflow as tf


def model(input_data):
    def conv_3d_pad_leakyReLU(input, name, kd, kh, kw, sd, sh, sw, n_out):
        n_in = input.get_shape()[4].value
        padding = tf.constant([[0, 0], [1, 1], [1, 1], [1, 1], [0, 0]])
        input_pad = tf.pad(tensor=input, paddings=padding, mode="CONSTANT")
        kernel = tf.get_variable(name=name, shape=[kd, kh, kw, n_in, n_out], dtype=tf.float32,
                                 initializer=tf.contrib.layers.xavier_initializer(dtype=tf.float32))
        conv = tf.nn.conv3d(input_pad, filter=kernel, strides=[1, sd, sh, sw, 1], padding="VALID", data_format="NDHWC")
        print conv.get_shape().as_list()
        relu = tf.nn.leaky_relu(conv)
        return relu

    def conv_3d_pad_bn_leakyReLU(input, name, kd, kh, kw, sd, sh, sw, n_out):
        n_in = input.get_shape()[4].value
        padding = tf.constant([[0, 0], [1, 1], [1, 1], [1, 1], [0, 0]])
        input_pad = tf.pad(tensor=input, paddings=padding, mode="CONSTANT")
        kernel = tf.get_variable(name=name, shape=[kd, kh, kw, n_in, n_out], dtype=tf.float32,
                                 initializer=tf.contrib.layers.xavier_initializer(dtype=tf.float32))
        conv = tf.nn.conv3d(input_pad, filter=kernel, strides=[1, sd, sh, sw, 1], padding="VALID", data_format="NDHWC")
        print conv.get_shape().as_list()
        norm = tf.layers.batch_normalization(conv)
        relu = tf.nn.leaky_relu(norm)
        return relu

    def conv_3d_bn_leakyReLU(input, name, kd, kh, kw, sd, sh, sw, n_out):
        n_in = input.get_shape()[4].value
        kernel = tf.get_variable(name=name, shape=[kd, kh, kw, n_in, n_out], dtype=tf.float32,
                                 initializer=tf.contrib.layers.xavier_initializer(dtype=tf.float32))
        conv = tf.nn.conv3d(input, filter=kernel, strides=[1, sd, sh, sw, 1], padding="VALID", data_format="NDHWC")
        print conv.get_shape().as_list()
        norm = tf.layers.batch_normalization(conv)
        relu = tf.nn.leaky_relu(norm)
        return relu

    def deconv_bn_3d_relu(input, name, kd, kh, kw, sd, sh, sw, n_out):
        n_in = input.get_shape()[4].value
        n_out_dhw = input.get_shape()[1].value * 2
        kernel = tf.get_variable(name=name, shape=[kd, kh, kw, n_out, n_in], dtype=tf.float32,
                                 initializer=tf.contrib.layers.xavier_initializer(dtype=tf.float32))
        deconv = tf.nn.conv3d_transpose(value=input, filter=kernel,
                                        output_shape=[64, n_out_dhw, n_out_dhw, n_out_dhw, n_out],
                                        strides=[1, sd, sh, sw, 1], data_format="NDHWC")
        print deconv.get_shape().as_list()
        norm = tf.layers.batch_normalization(deconv)
        relu = tf.nn.relu(norm)
        return relu

    def deconv_bn_3d_relu_1(input, name, kd, kh, kw, sd, sh, sw, n_out):
        n_in = input.get_shape()[4].value
        n_out_dhw = input.get_shape()[1].value * 4
        kernel = tf.get_variable(name=name, shape=[kd, kh, kw, n_out, n_in], dtype=tf.float32,
                                 initializer=tf.contrib.layers.xavier_initializer(dtype=tf.float32))
        deconv = tf.nn.conv3d_transpose(value=input, filter=kernel,
                                        output_shape=[64, n_out_dhw, n_out_dhw, n_out_dhw, n_out],
                                        strides=[1, sd, sh, sw, 1], padding="VALID",data_format="NDHWC")
        print deconv.get_shape().as_list()
        norm = tf.layers.batch_normalization(deconv)
        relu = tf.nn.relu(norm)
        return relu

    def deconv_3d(input, name, kd, kh, kw, sd, sh, sw, n_out):
        n_in = input.get_shape()[4].value
        n_out_dhw = input.get_shape()[1].value * 2
        kernel = tf.get_variable(name=name, shape=[kd, kh, kw, n_out, n_in], dtype=tf.float32,
                                 initializer=tf.contrib.layers.xavier_initializer(dtype=tf.float32))
        deconv = tf.nn.conv3d_transpose(value=input, filter=kernel,
                                        output_shape=[64, n_out_dhw, n_out_dhw, n_out_dhw, n_out],
                                        strides=[1, sd, sh, sw, 1], data_format="NDHWC")
        #relu=tf.nn.relu(deconv)

        print deconv.get_shape().as_list()
        return deconv

    # 2*32^3
    conv_1 = conv_3d_pad_leakyReLU(input=input_data, name="conv_1", kd=4, kh=4, kw=4, sd=2, sh=2, sw=2, n_out=80)
    # 80*16^3
    conv_2 = conv_3d_pad_bn_leakyReLU(input=conv_1, name="conv_2", kd=4, kh=4, kw=4, sd=2, sh=2, sw=2, n_out=160)
    # 160*8^3
    conv_3 = conv_3d_pad_bn_leakyReLU(input=conv_2, name="conv_3", kd=4, kh=4, kw=4, sd=2, sh=2, sw=2, n_out=320)
    # 320*4^3
    conv_4 = conv_3d_bn_leakyReLU(input=conv_3, name="conv_4", kd=4, kh=4, kw=4, sd=1, sh=1, sw=1, n_out=640)
    # 640*1^3
    reshape_3 = tf.reshape(tensor=conv_4, shape=[-1, 640])
    # 640
    w_fc1 = tf.get_variable(name="weight_1", shape=[640, 640], dtype=tf.float32,
                            initializer=tf.truncated_normal_initializer(stddev=0.1))
    b_fc1 = tf.get_variable(name="bias_1", shape=[640], dtype=tf.float32, initializer=tf.constant_initializer(0.1))
    fc1 = tf.nn.relu(tf.matmul(reshape_3, w_fc1) + b_fc1)
    print fc1.get_shape().as_list()
    # 512
    w_fc2 = tf.get_variable(name="weight_2", shape=[640, 640], dtype=tf.float32,
                            initializer=tf.truncated_normal_initializer(stddev=0.1))
    b_fc2 = tf.get_variable(name="bias_2", shape=[640], dtype=tf.float32, initializer=tf.constant_initializer(0.1))
    fc2 = tf.nn.relu(tf.matmul(fc1, w_fc2) + b_fc2)
    print fc2.get_shape().as_list()
    # 640
    reshape_fc2 = tf.reshape(tensor=fc2, shape=[-1, 1, 1, 1, 640])
    print reshape_fc2.get_shape().as_list()
    d_1 = tf.concat([reshape_fc2, conv_4], 4)
    # 1280*1^3
    deconv_1 = deconv_bn_3d_relu_1(input=d_1, name="deconv_1", kd=4, kh=4, kw=4, sd=1, sh=1, sw=1, n_out=320)
    d_2 = tf.concat([deconv_1, conv_3], 4)
    # 640*4^3
    deconv_2 = deconv_bn_3d_relu(input=d_2, name="deconv_2", kd=4, kh=4, kw=4, sd=2, sh=2, sw=2, n_out=160)
    d_3 = tf.concat([deconv_2, conv_2], 4)
    # 320*8^3
    deconv_3 = deconv_bn_3d_relu(input=d_3, name="deconv_3", kd=4, kh=4, kw=4, sd=2, sh=2, sw=2, n_out=80)
    d_4 = tf.concat([deconv_3, conv_1], 4)
    # 160*16^3
    result = deconv_3d(input=d_4, name="result", kd=4, kh=4, kw=4, sd=2, sh=2, sw=2, n_out=1)
    constant=tf.cast(1,dtype=tf.float32)
    result=tf.log(tf.add(tf.abs(result),constant))
    return result