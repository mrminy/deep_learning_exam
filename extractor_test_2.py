#!/usr/bin/env python

import tensorflow as tf
import numpy as np

from data_fetcher import load_data

batch_size = 32
test_size = 64


def init_weights(shape):
    return tf.Variable(tf.random_normal(shape, stddev=0.01))


def model(X, w, w2, w3, w4, w_o, p_keep_conv, p_keep_hidden):
    l1a = tf.nn.relu(tf.nn.conv2d(X, w,  # l1a shape=(?, 28, 28, 32)
                                  strides=[1, 1, 1, 1], padding='SAME'))
    l1 = tf.nn.max_pool(l1a, ksize=[1, 2, 2, 1],  # l1 shape=(?, 14, 14, 32)2
                        strides=[1, 2, 2, 1], padding='SAME')
    l1 = tf.nn.dropout(l1, p_keep_conv)

    l2a = tf.nn.relu(tf.nn.conv2d(l1, w2,  # l2a shape=(?, 14, 14, 64)
                                  strides=[1, 1, 1, 1], padding='SAME'))
    l2 = tf.nn.max_pool(l2a, ksize=[1, 2, 2, 1],  # l2 shape=(?, 7, 7, 64)
                        strides=[1, 2, 2, 1], padding='SAME')
    l2 = tf.nn.dropout(l2, p_keep_conv)

    l3a = tf.nn.relu(tf.nn.conv2d(l2, w3,  # l3a shape=(?, 7, 7, 128)
                                  strides=[1, 1, 1, 1], padding='SAME'))
    l3 = tf.nn.max_pool(l3a, ksize=[1, 2, 2, 1],  # l3 shape=(?, 4, 4, 128)
                        strides=[1, 2, 2, 1], padding='SAME')
    l3 = tf.reshape(l3, [-1, w4.get_shape().as_list()[0]])  # reshape to (?, 2048)
    l3 = tf.nn.dropout(l3, p_keep_conv)

    l4 = tf.nn.relu(tf.matmul(l3, w4))
    l4 = tf.nn.dropout(l4, p_keep_hidden)

    pyx = tf.matmul(l4, w_o)
    pyx = tf.nn.softmax(pyx)
    return pyx


training_data, trY = load_data()
teX, teY = load_data(data_path='test_images.npy', labels_path='test_one_hot.npy')

trX = training_data.reshape(-1, 256, 192, 3)  # 28x28x1 input img
teX = teX.reshape(-1, 256, 192, 3)  # 28x28x1 input img

X = tf.placeholder("float", [None, 256, 192, 3])
Y = tf.placeholder("float", [None, 16825])

w = init_weights([3, 3, 3, 32])  # 3x3x1 conv, 32 outputs
w2 = init_weights([3, 3, 32, 64])  # 3x3x32 conv, 64 outputs
w3 = init_weights([3, 3, 64, 128])  # 3x3x32 conv, 128 outputs
w4 = init_weights([128 * 4 * 4 * 48, 6000])  # FC 128 * 4 * 4 inputs, 625 outputs
w_o = init_weights([6000, 16825])  # FC 625 inputs, 10 outputs (labels)

p_keep_conv = tf.placeholder("float")
p_keep_hidden = tf.placeholder("float")
py_x = model(X, w, w2, w3, w4, w_o, p_keep_conv, p_keep_hidden)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(py_x, Y))
train_op = tf.train.AdamOptimizer(learning_rate=0.001).minimize(cost)
# train_op = tf.train.RMSPropOptimizer(0.001, 0.9).minimize(cost)

correct_pred = tf.pow((Y - py_x), 2)
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Launch the graph in a session
config = tf.ConfigProto(log_device_placement=True, device_count={'GPU': 0})
with tf.Session(config=config) as sess:
    # you need to initialize all variables
    tf.initialize_all_variables().run()

    for i in range(10):
        # training_batch = zip(range(0, len(trX), batch_size),
        #                      range(batch_size, len(trX) + 1, batch_size))
        idexes = np.arange(len(training_data))
        for _ in range(5):
            idx = np.random.choice(idexes, batch_size, replace=True)
            batch_x = trX[idx]
            batch_y = trY[idx]
            cost = sess.run(train_op, feed_dict={X: batch_x, Y: batch_y, p_keep_conv: 0.8, p_keep_hidden: 0.5})

        test_indices = np.arange(len(teX))  # Get A Test Batch
        np.random.shuffle(test_indices)
        test_indices = test_indices[0:test_size]

        print(i, "Cost:", cost, "Acc:", sess.run(accuracy,
                                                 feed_dict={X: teX[test_indices], Y: teY[test_indices],
                                                            p_keep_conv: 1.0,
                                                            p_keep_hidden: 1.0}))

    test_indices = np.arange(len(teX))  # Get A Test Batch
    np.random.shuffle(test_indices)
    test_indices = test_indices[0:test_size]
    print("Final acc:", sess.run(accuracy,
                                 feed_dict={X: teX[test_indices], Y: teY[test_indices], p_keep_conv: 1.0,
                                            p_keep_hidden: 1.0}))

    index = np.argmax(teY[0])
    print(teY[0][index])
    print(sess.run(py_x, feed_dict={X: [teX[0]], p_keep_conv: 1.0, p_keep_hidden: 1.0})[0][index])
