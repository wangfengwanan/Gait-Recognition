import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# build network
xs = tf.placeholder(tf.float32, [None, 561], "input")
ys = tf.placeholder(tf.float32, [None, 12], "output")


l1 = tf.layers.dense(xs, 1122, tf.nn.relu, name="l1")
l2 = tf.layers.dense(l1, 1122, tf.nn.relu, name="l2")
out = tf.layers.dense(l2, 12, name="l3")
prediction = tf.nn.softmax(out, name="pred")

cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(prediction),reduction_indices=[1]))    
accuracy = tf.metrics.accuracy(labels=tf.argmax(ys, axis=1), predictions=tf.argmax(out, axis=1),)[1]
train_step = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cross_entropy)

sess = tf.Session()
sess.run(tf.group(tf.global_variables_initializer(), tf.local_variables_initializer()))

for i in range(5000):
    batch_index = np.random.randint(len(x_train_datas), size=32)
    batch_xs, batch_ys = np.array(x_train_datas)[batch_index],np.array(y_train_datas)[batch_index]
    sess.run(train_step, feed_dict={xs: batch_xs, ys: batch_ys})
    if i % 50 == 0:
        print(sess.run(accuracy,feed_dict={xs:x_train_datas,ys:y_train_datas}))
