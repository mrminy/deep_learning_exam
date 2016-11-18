#!/usr/bin/env python
"""
https://github.com/nlintz/TensorFlow-Tutorials/blob/master/05_convolutional_net.py
"""

import tensorflow as tf
import numpy as np

from data_fetcher import load_data, convert_to_one_hot


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


class FeatureExtractor:
    def __init__(self, restore_path=None):
        self.batch_size = 64
        self.test_size = 64
        if restore_path is None:
            self.save_path = 'checkpoints4/'
        else:
            self.save_path = restore_path

        # Network variables
        self.X = None
        self.p_keep_conv = None
        self.p_keep_hidden = None
        self.sess = None
        self.py_x = None  # Prediction

    def generate_features(self, img):
        """
        Give output for an image
        :param img:
        :return:
        """
        # assert img correct resolution
        img = img.reshape(-1, 64, 48, 3)
        result = self.sess.run(self.py_x, feed_dict={self.X: img, self.p_keep_conv: 1.0, self.p_keep_hidden: 1.0})
        return result[0]

    def train(self, train=True):
        self.X = tf.placeholder("float", [None, 64, 48, 3])
        Y = tf.placeholder("float", [None, 16825])

        w = init_weights([3, 3, 3, 32])  # 3x3x1 conv, 32 outputs
        w2 = init_weights([3, 3, 32, 64])  # 3x3x32 conv, 64 outputs
        w3 = init_weights([3, 3, 64, 128])  # 3x3x32 conv, 128 outputs
        w4 = init_weights([128 * 4 * 4 * 3, 6000])  # FC 128 * 4 * 4 inputs, 625 outputs
        w_o = init_weights([6000, 16825])  # FC 625 inputs, 10 outputs (labels)

        self.p_keep_conv = tf.placeholder("float")
        self.p_keep_hidden = tf.placeholder("float")
        self.py_x = model(self.X, w, w2, w3, w4, w_o, self.p_keep_conv, self.p_keep_hidden)

        # cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(self.py_x, Y))
        # cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(self.py_x, Y))
        cost = -tf.reduce_sum(((Y * tf.log(self.py_x + 1e-9)) + ((1 - Y) * tf.log(1 - self.py_x + 1e-9))))
        # cost = tf.reduce_mean(tf.square(Y - self.py_x))
        train_op = tf.train.AdamOptimizer(learning_rate=0.001).minimize(cost)
        # train_op = tf.train.RMSPropOptimizer(0.001, 0.9).minimize(cost)

        correct_pred = tf.pow((Y - self.py_x), 2)
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

        saver = tf.train.Saver()

        # Launch the graph in a session
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        # config = tf.ConfigProto(log_device_placement=True, device_count={'GPU': 0})
        self.sess = tf.Session(config=config)
        # you need to initialize all variables
        self.sess.run(tf.initialize_all_variables())

        if train:
            # Load training and validation data
            trX, trY = load_data(data_path='data/small_size/train_images_small.npy',
                                 labels_path='data/small_size/train_one_hot.npy')
            teX, teY = load_data(data_path='data/small_size/test_images_small.npy',
                                 labels_path='data/small_size/test_one_hot.npy')
            trX = trX.reshape(-1, 64, 48, 3)  # 28x28x1 input img
            teX = teX.reshape(-1, 64, 48, 3)  # 28x28x1 input img

            for i in range(200):
                training_batch = zip(range(0, len(trX), self.batch_size),
                                     range(self.batch_size, len(trX) + 1, self.batch_size))
                print("Batches:", int(len(trX) / self.batch_size))
                batch_counter = 0
                total_cost = 0.0
                for start, end in training_batch:
                    batch_x = trX[start:end]
                    batch_y = convert_to_one_hot(trY[start:end])
                    _, last_cost = self.sess.run([train_op, cost],
                                                 feed_dict={self.X: batch_x, Y: batch_y, self.p_keep_conv: 0.8,
                                                            self.p_keep_hidden: 0.5})
                    batch_counter += 1
                    total_cost += last_cost
                    if batch_counter % 100 == 0:
                        print("Training batch nr", batch_counter, "- sample:", start, "-", end, "- cost:", last_cost)

                test_indices = np.arange(len(teX))  # Get a test batch
                np.random.shuffle(test_indices)
                test_indices = test_indices[0:self.test_size]
                test_batch_x = teX[test_indices]
                test_batch_y = convert_to_one_hot(teY[test_indices])
                print("Epoch", i, "- Total cost:", total_cost, "Acc:", self.sess.run(accuracy,
                                                                                     feed_dict={
                                                                                         self.X: test_batch_x,
                                                                                         Y: test_batch_y,
                                                                                         self.p_keep_conv: 1.0,
                                                                                         self.p_keep_hidden: 1.0}))

                # Saving model
                saver.save(self.sess, self.save_path + 'feature_extractor_model.ckpt', global_step=i + 1)

            test_indices = np.arange(len(teX))  # Get A Test Batch
            np.random.shuffle(test_indices)
            test_indices = test_indices[0:self.test_size]
            test_batch_x = teX[test_indices]
            test_batch_y = convert_to_one_hot(teY[test_indices])
            print("Finished acc", self.sess.run(accuracy, feed_dict={self.X: test_batch_x, Y: test_batch_y,
                                                                     self.p_keep_conv: 1.0,
                                                                     self.p_keep_hidden: 1.0}))

            index = np.argmax(teY[0])
            print(teY[0][index])
            print(
                self.sess.run(self.py_x, feed_dict={self.X: [teX[0]], self.p_keep_conv: 1.0, self.p_keep_hidden: 1.0})[
                    0][index])
        else:
            ckpt = tf.train.get_checkpoint_state(self.save_path)
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(self.sess, ckpt.model_checkpoint_path)
                print("Loaded model successfully")
            else:
                print("No model found, exiting...")
                self.close()
                exit()

    def close(self):
        self.sess.close()


if __name__ == '__main__':
    extractor = FeatureExtractor()
    extractor.train(train=True)
    print("Done!")
