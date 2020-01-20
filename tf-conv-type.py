import tensorflow as tf
import cv2
import numpy as np
print(tf.__version__)


kernel_size   = 3
kernel_out = 12
strides = (1,1)
#-------------------------------------
# tf.nn,  most complex and old
def func_nn(input_node):
	with tf.variable_scope('nn'):
		#input_node = tf.transpose(input_node, [1, 2, 3, 0]) # (1,13,13,1) -> (13,13,1,1)

		print('input_node.shape', input_node.shape)
		w = tf.get_variable(name='weight', shape=[kernel_size, kernel_size, 3, kernel_out], dtype=tf.float32)
		b = tf.get_variable(name='bias', shape=[kernel_out], dtype=tf.float32)
		out = tf.nn.conv2d(input_node, filter=w, strides=[1,1,1,1], padding='SAME') 
		# conv2d (input, filter, strides, )
		# filter = (h, w, in_ch, out_ch) == w

		out = out + b
	return out

# easy, good for testing
def func_layer(input_node):
	out = tf.layers.conv2d(input_node, kernel_out, kernel_size, strides=strides, padding='same', name='layer')
	return out

def func_slim(input_node):
	out = tf.contrib.slim.conv2d(input_node, kernel_out, kernel_size, stride=strides, padding='SAME', activation_fn=None, scope='slim')
	return out

def func_keras(input_node):
	model = tf.keras.Sequential([tf.keras.layers.Conv2D(kernel_out, kernel_size, strides=strides, padding='same')], name = 'keras')
	return model(input_node)

node = tf.placeholder(shape=[None, 100,100,3], dtype=tf.float32)

out_nn    = func_nn(node)
out_layer = func_layer(node)
out_slim  = func_slim(node)
out_keras = func_keras(node)

img = cv2.imread('dog.jpg')
img = cv2.resize(img, (100,100)) 

img = np.expand_dims(img, 0) # (1,100,100,3)

tf.summary.FileWriter("board", graph=tf.get_default_graph())

with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	sess.run(tf.local_variables_initializer())

	nn_result, layer_result = sess.run([out_nn, out_layer], feed_dict={node:img})
