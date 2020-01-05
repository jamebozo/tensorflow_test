import tensorflow as tf

board_path = './board'
def main():
	""" Test variable, get_variable, name_scope, and variable_scope 
	"""
	input_node = tf.placeholder(tf.int32)
	const = tf.constant(1, tf.int32, name='constant')
	tf.add(input_node, const, name='result')	

	#----------------------------
	# same
	with tf.name_scope('ns1'):
		tf.placeholder(tf.int32, name='i1')

	with tf.variable_scope('vs1'):
		tf.placeholder(tf.int32, name='i2')

	# duplicate name -> V_0, V_1
	with tf.name_scope('ns2'):
		tf.Variable(name = 'V', initial_value=0)
		tf.Variable(name = 'V', initial_value=0)
		
	# test name_scope vs get_variable
	with tf.name_scope('weights'):
		a = tf.Variable(name='a1', initial_value=[1,2,3])
		b = tf.get_variable('b1',[1,2,3])

	with tf.Session() as sess:
		print(a)
		print(b)

	tf.summary.FileWriter(board_path, graph=tf.get_default_graph())


if __name__ == '__main__':
	main()