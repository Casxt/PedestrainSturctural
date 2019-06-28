import tensorflow as tf
c = tf.constant([1.0, 2.0])
d = tf.constant([0.0, 1.0])
e = tf.concat([c[0:1], d[1:]], 0)
