import tensorflow as tf

board_path = './board'
def main():
	""" Test variable, get_variable, name_scope, and variable_scope 
	"""
	input_node = tf.placeholder(tf.int32)
	const = tf.constant(1, tf.int32, name='constant')
	tf.add(input_node, const, name='result')	

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