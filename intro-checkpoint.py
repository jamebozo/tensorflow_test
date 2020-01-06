import tensorflow as tf
import numpy as np

def get_first_filters(sess):
	tensor = tf.get_default_graph().get_tensor_by_name("conv_1/kernel/read:0")
	return sess.run(tensor[1,:,:,1])

def save():

	input_node = tf.placeholder(shape=[None, 100, 100, 3], dtype=tf.float32)
	net = tf.layers.conv2d(input_node, 32, (3,3), strides=(2,2), padding='same', name='conv_1') # in, out, ksize, stride, padding
	net = tf.layers.conv2d(net, 32, (3,3), strides=(1,1), padding='same', name='conv_2') # in, out, ksize, stride, padding
	net = tf.layers.conv2d(net, 64, (3,3), strides=(2,2), padding='same', name='conv_3') # in, out, ksize, stride, padding

	tf.summary.FileWriter("board", graph=tf.get_default_graph())
	saver = tf.train.Saver()

	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		sess.run(tf.local_variables_initializer())
		

		result = get_first_filters(sess)

		saver.save(sess, "ckpt/model.ckpt")
		
	return result

def load():
    tf.reset_default_graph()
    with tf.Session() as sess:
    	saver = tf.train.import_meta_graph('ckpt/model.ckpt.meta') 
    	# If you restore first, initializer will re-init variables xxx
    	#    ex: saver.restore(sess, 'ckpt/model.ckpt')  
    	sess.run(tf.global_variables_initializer())
    	sess.run(tf.local_variables_initializer())

    	saver.restore(sess, 'ckpt/model.ckpt')
    	result = get_first_filters(sess)
    return result

if __name__ == '__main__':
	save_value = save()
	load_value = load()

	print(save_value)
	print(load_value)

	assert np.alltrue(save_value == load_value)