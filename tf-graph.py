import tensorflow as tf
import numpy as np

# for ii in range(3):
# 	a = tf.constant(0, name='a')

# #---------------------------------
# # `show whole graph`
# print(tf.get_default_graph().as_graph_def())
# # node {
# #   name: "a"
# #   op: "Const"
# #   attr {
# # }
# # node {
# #   name: "a_1"
# #   ....
# # }


# #---------------------------------
# # duplicate naming
# print([n.name for n in tf.get_default_graph().as_graph_def().node])
# # out: ['a', 'a_1', 'a_2']


#---------------------------------
# multiple graphs
tf.constant(0, name="a0")
# as_default() defines everything underneath as g1's nodes 
with tf.Graph().as_default() as g1:  
	tf.constant(1, name="a1")
with tf.Graph().as_default() as g2:  
	tf.constant(2, name="a2")

sess0 = tf.Session()          # default graph
sess1 = tf.Session(graph=g1)  # sess1 only use g1 graph
sess2 = tf.Session(graph=g2)  # sess2 only use g2 graph

t0 = tf.get_default_graph().get_tensor_by_name('a0:0')
t1 = sess1.graph.get_tensor_by_name('a1:0')
t2 = sess2.graph.get_tensor_by_name('a2:0')

res0 = sess0.run(t0)
res1 = sess1.run(t1)
res2 = sess2.run(t2)

print("default: {}".format([n.name for n in tf.get_default_graph().as_graph_def().node]))
print("graph 1: {}".format([n.name for n in g1.as_graph_def().node]))
print("graph 2: {}".format([n.name for n in g2.as_graph_def().node]))