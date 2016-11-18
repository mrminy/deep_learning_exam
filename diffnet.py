import pickle

import math
import numpy as np
import time
from matplotlib import pyplot as plt
import tensorflow as tf
import imgnet_test
from data_fetcher import load_image, find_img_path
# from extractor_test_2 import FeatureExtractor
from front import score


class DiffNet:
    def __init__(self, db):
        # self.feature_extractor = FeatureExtractor(restore_path='checkpoints4/')
        # self.feature_extractor.train(train=False)
        self.feature_extractor = imgnet_test.ImageFeatureExtractor()
        self.db = db
        print("Running feature extracting on db images")
        self.db_predictions = self.feature_extractor.run_inference_on_images(self.db)
        print("Finished extracting features from db")

        self.history_sampling_rate = 100
        self.display_step = 1

        # self.threshold = 0.000061

        self.graph = tf.Graph()

        # tf placeholders
        self.Q = None  # Query input
        self.T = None  # Test input (image from db)
        self.Y = None
        self.y_pred = None
        self.sess = None
        self.keep_prob = None  # for dropout
        self.saver = None

        # Network Parameters
        self.n_input = 1008 * 2  # 1008 image features
        self.n_output = 1  # Done (1) or not done (0)
        self.n_hidden_1 = 1000  # 1st layer num features
        self.n_hidden_2 = 1000  # 2nd layer num features

        self.cost_history = []
        self.test_acc_history = []

    def build(self, learning_rate=0.01):
        print("Building done graph...")
        with self.graph.as_default():
            # Encode curr_state, add transition prediction with selected action and decode to predicted output state
            self.Q = tf.placeholder("float", [None, self.n_input / 2])  # query placeholder
            self.T = tf.placeholder("float", [None, self.n_input / 2])  # test placeholder
            self.Y = tf.placeholder("float", [None, self.n_output])  # similarity prediction
            self.keep_prob = tf.placeholder(tf.float32)  # For dropout

            weights = {
                'h1': tf.Variable(
                    tf.random_normal([self.n_input, self.n_hidden_1])),
                'h2': tf.Variable(
                    tf.random_normal([self.n_hidden_1, self.n_hidden_2])),
                'out': tf.Variable(
                    tf.random_normal([self.n_hidden_2, self.n_output])),
            }
            biases = {
                'b1': tf.Variable(tf.random_normal([self.n_hidden_1])),
                'b2': tf.Variable(tf.random_normal([self.n_hidden_2])),
                'bout': tf.Variable(tf.random_normal([self.n_output])),
            }

            input_tensor = np.concatenate((self.Q, self.T), 0)
            layer_1 = tf.nn.tanh(tf.add(tf.matmul(input_tensor, weights['h1']), biases['b1']))
            layer_1_drop = tf.nn.dropout(layer_1, self.keep_prob)  # Dropout layer
            layer_2 = tf.nn.tanh(tf.add(tf.matmul(layer_1_drop, weights['h2']), biases['b2']))
            layer_2_drop = tf.nn.dropout(layer_2, self.keep_prob)  # Dropout layer
            out = tf.nn.tanh(tf.add(tf.matmul(layer_2_drop, weights['out']), biases['bout']))
            out = tf.nn.dropout(out, self.keep_prob)  # Dropout layer

            # Prediction
            self.y_pred = out
            # Targets (Labels) are the input data.
            y_true = self.Y

            # Define loss, minimize the squared error (with or without scaling)
            self.loss_function = tf.reduce_mean(tf.square(y_true - self.y_pred))
            self.optimizer = tf.train.AdamOptimizer(learning_rate).minimize(self.loss_function)
            # self.optimizer = tf.train.RMSPropOptimizer(learning_rate, momentum=0.01).minimize(self.loss_function)

            # Evaluate model
            self.accuracy = tf.reduce_mean(tf.cast(self.loss_function, tf.float32))

            # Creates a saver
            self.saver = tf.train.Saver()

            # Initializing the variables
            self.init = tf.initialize_all_variables()

            # Launch the graph
            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            self.sess = tf.Session(graph=self.graph, config=config)
            self.sess.run(self.init)

    def train(self, training_epochs=20, learning_rate=0.001, batch_size=32, show_cost=False, show_test_acc=False,
              save=False, training_data=None, test_data_r=None, test_data_ac=None,
              save_path='diffnet1/', logger=True):
        # Load and preprocess data
        if logger:
            print("Loading and preprocessing data...")
        X_train, Y_train = self.preprocess_data(training_data)
        X_test_r, Y_test_r = self.preprocess_data(test_data_r)

        self.build(learning_rate=learning_rate)

        total_batch = int(len(X_train) / batch_size)
        if logger:
            print("Starting training...")
            print("Total nr of batches:", total_batch)
        # Training cycle
        for epoch in range(training_epochs):
            # Loop over all batches
            c = None
            idexes = np.arange(len(X_train))
            for i in range(total_batch):
                idx = np.random.choice(idexes, batch_size, replace=True)
                batch_xs = X_train[idx]
                batch_ys = Y_train[idx]
                # Run optimization op (backprop) and cost op (to get loss value)
                _, c = self.sess.run([self.optimizer, self.loss_function],
                                     feed_dict={self.Q: batch_xs, self.Y: batch_ys, self.keep_prob: 1.0})
                if i % self.history_sampling_rate == 0:
                    self.cost_history.append(c)
                    sampled_indexes = np.random.choice(np.arange(0, len(X_test_r)), 50000)
                    self.test_acc_history.append(self.sess.run(self.accuracy,
                                                               feed_dict={self.Q: X_test_r[sampled_indexes],
                                                                          self.Y: Y_test_r[sampled_indexes],
                                                                          self.keep_prob: 1.0}))

            # Display logs per epoch step
            if epoch % self.display_step == 0 and c is not None:
                test_error = self.sess.run(self.accuracy, feed_dict={self.Q: X_test_r[sampled_indexes],
                                                                     self.Y: Y_test_r[sampled_indexes],
                                                                     self.keep_prob: 1.0})
                print("Epoch:", '%04d' % (epoch + 1), "cost=", "{:.9f}".format(c), "test error=",
                      "{:.9f}".format(test_error))

        acc_random_agent = self.sess.run(self.accuracy,
                                         feed_dict={self.Q: X_test_r, self.Y: Y_test_r,
                                                    self.keep_prob: 1.0})
        print("Final test error random agent:", acc_random_agent)

        # Applying encode and decode over test set and show some examples
        # output = self.sess.run(self.y_pred,feed_dict={self.X: X_test_r[:self.examples_to_show], self.keep_prob: 1.0})
        # print(Y_test_r[:self.examples_to_show])
        # print(output[:])

        if save:
            save_path = self.saver.save(self.sess, save_path)
            print("Model saved in file: %s" % save_path)

        if show_test_acc:
            y_axis = np.array(self.test_acc_history)
            plt.plot(y_axis)
            plt.show()

        if show_cost:
            y_axis = np.array(self.cost_history)
            plt.plot(y_axis)
            plt.show()

    def query(self, query_img_name):
        path = 'validate/pics/*/'
        # q_img_path = find_img_path(path, query_img_name)
        # q_img = load_image(query_img_name, path=path)
        # query_features = self.feature_extractor.generate_features(q_img)
        query_features = self.feature_extractor.run_inference_on_image(query_img_name, path=path)
        if query_features is None:
            return None
        query_features_indexes = np.where(query_features > query_features.max() * 0.5)
        equality_threshold = len(query_features_indexes[0]) * 0.5
        print("Equality threshold:", equality_threshold)
        cluster = []

        for i, pred in enumerate(self.db_predictions):
            if pred is not None and self.db[i] != query_img_name:
                db_features_indexes = np.where(pred > pred.max() * 0.5)
                nr_of_equal_features = np.sum(query_features_indexes[0] == db_features_indexes[0])
                if nr_of_equal_features > equality_threshold:
                    cluster.append(self.db[i])

                if i % 1000 == 0:
                    print(i, len(cluster))

                # TODO find diff in features and select equal images only
                # mse = ((query_features - db_features) ** 2).mean()
                # if mse < self.threshold:
                #     cluster.append(db_img_name)

                if len(cluster) >= 50:
                    print("Breaks because of big cluster...")
                    break
            return cluster
        return None


if __name__ == '__main__':
    start_time = time.time()
    training_labels = pickle.load(open('./validate/pickle/combined.pickle', 'rb'))
    training_imgs = list(training_labels.keys())
    net = DiffNet(training_imgs)
    scores = []
    for j in range(len(training_imgs)):
        query_img = training_imgs[j]
        cluster = net.query(query_img)
        if cluster is not None and len(cluster) > 0:
            score_res = score(training_labels, target=query_img, selection=cluster, n=len(cluster))
            scores.append(score_res)
            print(j, "\t\t", query_img, "- cluster size:", len(cluster), "- score", score_res)
        else:
            scores.append(0.0)
            print(j, "\t\t", query_img, "- No similar images found...")
    print("Average over 100:", np.mean(np.array(scores)))
    print("Time used:", time.time() - start_time)