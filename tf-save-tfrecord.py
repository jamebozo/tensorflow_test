import tensorflow as tf
name_list = ['A', 'B']
img_list = ['a.jpg', 'b.jpg']
age_list = [12,33]
hgt_list = [140.2, 174.7]
prf_list = [[1,2],[1,5]]

writer = tf.python_io.TFRecordWriter('tfrecord/member.tfrecord')

for i, name in enumerate(name_list):
	member_name = name.encode('utf8')
	img = img_list[i]
	age = age_list[i]
	hgt = hgt_list[i]
	prf = prf_list[i]

	with tf.gfile.GFile(os.path.join('my-icons', img), 'rb' ) as fid:
		encode_image = fid.read()

	# tf record is measured in `examples`:
	# an example is a dict of lists, even if only 1 elem (1 elem list)

	def bytes_feature(value):
		return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

	def float_feature(value):
		return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

	def data_to_example(name, encoded_img, age, hgt, prf):
		example = tf.train.Example(features=tf.train.Features(feature={
			'member/name':bytes_feature(name),
			'member/age': bytes_feature(age),
			'member/hight':bytes_feature(hgt),
			'member/prefer':bytes_feature(prf),
			}))
	return example

	writer.write(tf.example.SerializeToString)