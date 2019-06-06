import tensorflow as tf
from darknet import Darknet53

core_obj = Darknet53(darknet53_npz_path="darknet53.conv.74.npz", trainable=True, scratch=True)
start = tf.placeholder(tf.float32,shape=(None,640, 480, 3))
core = core_obj.build(start, tf.constant(True, tf.bool) )
with tf.Session() as sess:
    writer = tf.summary.FileWriter("log/", sess.graph)
    writer.close()
