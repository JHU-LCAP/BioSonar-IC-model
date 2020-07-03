import tensorflow as tf

def arcsinh(x):
	return tf.math.log(tf.math.add(tf.math.sqrt(tf.math.add(tf.math.square(x),1)), x))

def bent_identity(x):
	return tf.math.add(tf.math.sqrt(tf.math.add(tf.math.square(x),1))/2,x)

def conv2d(layer_name, x, nfilter, filter_size, step):
    return tf.layers.conv2d(inputs=x, filters=nfilter, kernel_size=filter_size, strides=step, padding='SAME', name=layer_name)

def conv2d_with_filter(layer_name, x, filters, sliding, skip):
	return tf.nn.conv2d(input=x, filter=filters, strides=sliding, dilations= skip, padding='SAME', name=layer_name)

def conv2d_transpose(layer_name, x, kernel_size, num_outputs, step):
    return tf.layers.conv2d_transpose(inputs=x, filters=num_outputs, kernel_size=kernel_size, strides =step, padding='SAME', name=layer_name)

def conv2d_trans_with_filter(layer_name, x, filters, out_shape, step):
	return tf.nn.conv2d_transpose(value=x, filter=filters, output_shape=out_shape, strides=step, padding='SAME', name=layer_name)

def maxpool2d(x, pool_size, step):
    return tf.layers.max_pooling2d(inputs=x, pool_size=pool_size, strides=step, padding='SAME')
