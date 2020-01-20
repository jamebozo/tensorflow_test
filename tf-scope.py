import tensorflow as tf

board_path = './board'
def main():
	""" Test variable, get_variable, name_scope, and variable_scope 
	"""
	input_node = tf.placeholder(tf.int32)
	const = tf.constant(1, tf.int32, name='constant')
	tf.add(input_node, const, name='result')	

	#----------------------------
	# usually, name_scope and varaible_scope show the same
	with tf.name_scope('ns1'):
		tf.placeholder(tf.int32, name='i1')
	with tf.variable_scope('vs1'):
		tf.placeholder(tf.int32, name='i2')

	#----------------------------
	# but if you have get_variable, ns22 will be `out of scope`
	with tf.name_scope('ns2'):
		# Both Inits new Variables
		ns21 = tf.Variable(name = 'ns21', initial_value=0)
		ns22 = tf.get_variable('ns21',[1,2,3])   

	# That's because name_scope doen't check name, 
	# Better practice: use `variable_scope` and `get_variable`
	with tf.variable_scope('nv3'):
		ns31 = tf.get_variable( 'nv31', [1,2,3])
		# ns32 = tf.get_variable('ns31', [1,2,3])  # ValueError:  Did you mean to set reuse=True or reuse=tf.AUTO_REUSE in VarScope

	# use "Reuse" if share same var
	with tf.variable_scope('nv4', reuse=tf.AUTO_REUSE):
		ns41 = tf.get_variable( 'nv41', [1,2,3])
		ns42 = tf.get_variable( 'nv41', [1,2,3])

	with tf.variable_scope('nv5') as scope:  # another way
		ns41 = tf.get_variable( 'nv51', [1,2,3])
		scope.reuse_variables()
		ns42 = tf.get_variable( 'nv51', [1,2,3])

	

	# test name_scope vs get_variable
	with tf.name_scope('weights'):
		a = tf.Variable(name='a1', initial_value=[1,2,3])
		b = tf.get_variable('b1',[1,2,3])

	with tf.Session() as sess:
		print(a)
		print(b)
		print(ns21)
		print(ns22)

	tf.summary.FileWriter(board_path, graph=tf.get_default_graph())


if __name__ == '__main__':
	main()