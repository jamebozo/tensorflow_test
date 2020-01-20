import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
import tensorflow.keras as keras
import random

lr = 0.001
steps = 2000
batch_size = 128

input_size = 28*28 # 28*28
class_num = 10 
dropout = 0.5

class_name = [str(i) for i in range(10)]
(train_x, train_y), (test_x, test_y) = mnist.load_data()

train_x = train_x / 255.0


#-----------------------
# show image samples
show_samples = False
if show_samples:
	plt.figure(figsize = (14, 14))
	for i in range(16):
		plt.subplot(4, 4, i+1) # 1....25
		plt.xticks([])          # no xticks, yticks
		plt.yticks([])
		plt.xlabel(class_name[train_y[i]])
		plt.imshow(train_x[i])
	plt.show()

print("train_x shape: {},  train_y shape {}".format(train_x.shape, train_y.shape))

#--------------------------
def keras_model():
	model = keras.Sequential(
		[keras.layers.Flatten(input_shape=(28, 28)), 
		 keras.layers.Dense(128, activation='relu'),
		 keras.layers.Dense(10, activation='softmax')
		])
	model.compile(optimizer='adam',
				  loss='sparse_categorical_crossentropy',
				  metrics=['accuracy'])
	model.fit(train_x, train_y, epochs=10)

# keras_model()

#--------------------------
def tf_model():
	init = tf.truncated_normal_initializer(dtype=tf.float32, stddev=1e-1)
	class net():
		with tf.variable_scope('conv1'):
			w1 = tf.get_variable("weights", [5,5,1,32], initializer=init, dtype=tf.float32)
			b1 = tf.get_variable("bias",    [32], initializer=init, dtype=tf.float32)
			tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME')

def tf_layer_model():
	x = tf.placeholder(tf.float32, [None, 28*28])/255.
	img = tf.reshape(x, [-1, 28, 28, 1])
	y = tf.placeholder(tf.int32, [None, 10])
	print('img.shape', img.shape)
	print('x shape {}, y shape {}'.format(x.shape, y.shape))
	
	conv1 = tf.layers.conv2d(inputs=img, filters=16, kernel_size=5, strides=1, padding='same', activation=tf.nn.relu)	
	print('conv1.shape', conv1.shape)  # (28, 28, 16)
	
	pool1 = tf.layers.max_pooling2d(conv1, pool_size=2, strides=2)
	print('pool1.shape', pool1.shape)  # (14, 14, 16)


	conv2 = tf.layers.conv2d(inputs=pool1, filters=32, kernel_size=5, strides=1, padding='same', activation=tf.nn.relu)	
	print('conv2.shape', conv2.shape)  # (14, 14, 32)
	
	pool2 = tf.layers.max_pooling2d(conv2, pool_size=2, strides=2)
	print('pool2.shape', pool2.shape)  # (7, 7, 32)

	flat = tf.reshape(pool2, [-1, 7*7*32])
	output = tf.layers.dense(flat, 10) # y hat
	print('output.shape', output.shape)

	#----------------
	loss = tf.losses.softmax_cross_entropy(onehot_labels=y, logits=output)
	optim = tf.train.AdamOptimizer(0.002).minimize(loss)

	accuracy = tf.metrics.accuracy(labels=tf.argmax(y, axis=1), predictions=tf.argmax(output, axis=1))
	#print('train_x.shape', train_x.shape)
	#print('train_y.shape', train_y.shape)

	#y2 = np.expand_dims(train_y, axis=1)
	#y2 = train_y[:, np.newaxis, np.newaxis]
	#print('train_y.shape', y2.shape)
	#np.stack((train_x, y2), axis=0)
	#print(train_x.shape)
	#res = np.concatenate((train_x, train_y), 0)
	#res = np.vstack((train_x, train_y))  # this won't work
	#----------------
	# shuffle data
	#data = [(_x, _y) for _x, _y in zip(train_x, train_y)]
	#print(data[:2])
	train_data = []
	for _x, _y in zip(train_x, train_y):
		train_data.append((_x/256., _y))
		#print(x.shape, y)
	np.array(train_data)

	test_data = []
	for _x, _y in zip(test_x, test_y):
		test_data.append((_x/256., _y))
		#print(x.shape, y)
	np.array(test_data)
	#random.shuffle(data)
	#print('data.shape', train_data)
	
	#----------------
	sess = tf.Session()
	init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer()) # the local var is for accuracy_op
	sess.run(init_op)     # initialize var in graph

	#---------------	
	for step in range(6000):  # 600*50 
		#print(data[:3])
		#print('first...')
		batch_list = random.sample(train_data, batch_size)
		#print('batch_list', batch_list[:3])
		#print('start unzip')
		bx, by = list(zip(*batch_list))
		#print(bx, by)
		b_x, b_y = np.array(bx), np.array(by)
		
		#print(b_x.shape, b_y.shape)

		#np.multiply.reduceat(b_x, [1,2], axis = 1)
		b_x = b_x.reshape(batch_size, -1)
		b_y = sess.run(tf.one_hot(b_y, 10, on_value=1, off_value=None))
		#print(b_x.shape, b_y.shape, type(b_x), type(b_y))	
		 
		#a = np.arange(8).reshape(2,2,2)
		#print(a, a.shape)
		#np.multiply.

		#print(x.shape, y.shape, type(x), type(y))

		sess.run(optim, feed_dict={x:b_x,  y:b_y})

		#print(res[:3].shape())
		#img, label = sth
		#
		#print(img.shape, label)
		#print(sth)
		#_, train_op = sess.run([], {x:bx, y:by})
		if step % 50 == 0:

			sess.run(optim, feed_dict={x:b_x, y:b_y})
			#test_x

			batch_list = random.sample(test_data, 100)
			bx, by = list(zip(*batch_list))
			#print(bx, by)
			b_x, b_y = np.array(bx), np.array(by)
			
			#print(b_x.shape, b_y.shape)

			#np.multiply.reduceat(b_x, [1,2], axis = 1)
			_test_x = b_x.reshape(100, -1)
			_test_y = sess.run(tf.one_hot(b_y, 10, on_value=1, off_value=None))

			loss_, accuracy_ = sess.run([loss, accuracy], {x: _test_x, y: _test_y})
			#print(loss_, accuracy_, np.argmax(accuracy_))
			print('Step:', step, '| train loss: %.4f' % loss_, '| test accuracy: %.2f' % np.argmax(accuracy_))

	#test_output = sess.run(output, {x, train_x[:10]})
	#pred_y = np.argmax(test_output, 1)
	#print("pred: {}  real: ".format(pred_y))

tf_layer_model()