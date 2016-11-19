import pickle
import random

import numpy as np
import time

from functools import reduce
from matplotlib import pyplot as plt
import tensorflow as tf
from imgnet_test import ImageFeatureExtractor
from data_fetcher import calculate_score_batch
from front import generate_dict_from_directory, score


class DiffNet:
    def __init__(self, db_path, test_db_path=None):
        self.feature_extractor = ImageFeatureExtractor()  # Inception-v3
        self.db_path = db_path
        self.test_db_path = test_db_path
        self.db = pickle.load(open('./' + db_path + '/pickle/combined.pickle', 'rb'))
        self.test_db = None
        if test_db_path is not None:
            self.test_db = pickle.load(open('./' + test_db_path + '/pickle/combined.pickle', 'rb'))
        self.db_features = None
        self.test_db_features = None

        self.history_sampling_rate = 5
        self.display_step = 1

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
        self.n_input = 2048 * 2  # 2048 features from second last layer in Inception-v3
        self.n_output = 1  # Equality score
        self.n_hidden_1 = 500  # 1st hidden layer
        self.n_hidden_2 = 50  # 2nd hidden layer

        self.model_name = 'diff_net.ckpt'
        self.cost_history = []
        self.test_acc_history = []

    def build(self, learning_rate=0.01):
        print("Building done graph...")
        with self.graph.as_default():
            # Encode curr_state, add transition prediction with selected action and decode to predicted output state
            self.Q = tf.placeholder("float", [None, int(self.n_input / 2)])  # query placeholder
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

            input_tensor = tf.concat(1, [self.Q, self.T])
            layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(input_tensor, weights['h1']), biases['b1']))
            layer_1_drop = tf.nn.dropout(layer_1, self.keep_prob)  # Dropout layer
            layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1_drop, weights['h2']), biases['b2']))
            layer_2_drop = tf.nn.dropout(layer_2, self.keep_prob)  # Dropout layer
            out = tf.nn.sigmoid(tf.add(tf.matmul(layer_2_drop, weights['out']), biases['bout']))
            out = tf.nn.dropout(out, self.keep_prob)  # Dropout layer

            # Prediction
            self.y_pred = out
            # Targets (Labels) are the input data.
            y_true = self.Y

            # Define loss, minimize the squared error (with or without scaling)
            self.loss_function = tf.reduce_mean(tf.square(y_true - self.y_pred))
            self.optimizer = tf.train.AdamOptimizer(learning_rate).minimize(self.loss_function)
            # self.optimizer = tf.train.RMSPropOptimizer(learning_rate, momentum=0.9).minimize(self.loss_function)

            # Evaluate model
            # self.accuracy = tf.reduce_mean(tf.cast(self.loss_function, tf.float32))

            # Creates a saver
            self.saver = tf.train.Saver()

            # Initializing the variables
            self.init = tf.initialize_all_variables()

            # Launch the graph
            # config = tf.ConfigProto()
            # config.gpu_options.allow_growth = True
            config = tf.ConfigProto(log_device_placement=True, device_count={'GPU': 0})
            self.sess = tf.Session(graph=self.graph, config=config)
            self.sess.run(self.init)

    def restore(self, path, global_step=1):
        if self.sess is None:
            self.build()
        self.saver.restore(self.sess, path + self.model_name + '-' + str(global_step))
        print("Restored Diffnet model")

    def train(self, training_epochs=20, learning_rate=0.001, batch_size=32, show_cost=False, show_test_acc=False,
              save=False, save_path='diffnet1/', logger=True):
        # Load and preprocess data
        if logger:
            print("Loading and preprocessing data...")
        X_train = np.array(list(self.db.keys()))
        X_test = np.array(list(self.test_db.keys()))

        if self.db_features is None:
            print("Running feature extracting on db images")
            self.db_features = self.feature_extractor.run_inference_on_images(self.db, path=self.db_path + '/pics/*/')
            print("Finished extracting features from db")
        if self.test_db_features is None:
            print("Running feature extracting on test db images")
            self.test_db_features = self.feature_extractor.run_inference_on_images(self.test_db,
                                                                                   path=self.test_db_path + '/pics/*/')
            print("Finished extracting features from test db")

        if self.sess is None:
            self.build(learning_rate=learning_rate)

        total_batch = int(len(X_train) / batch_size)
        if logger:
            print("Starting training...")
            print("Total nr of batches:", total_batch)
        # Training cycle
        training_pair_counter = 0
        for epoch in range(training_epochs):
            # Loop over all batches
            c = None
            idexes = np.arange(len(X_train))
            for i in range(total_batch):
                idx = np.random.choice(idexes, batch_size * 2, replace=True)
                q_idx = idx[:int(len(idx) / 2)]
                t_idx = idx[int(len(idx) / 2):]
                batch_qs = X_train[q_idx]  # Query images
                batch_ts = X_train[t_idx]  # Test images
                batch_ys = calculate_score_batch(batch_qs, batch_ts, self.db)
                batch_qs = self.db_features[q_idx]
                batch_ts = self.db_features[t_idx]
                # Run optimization op (backprop) and cost op (to get loss value)
                _, c = self.sess.run([self.optimizer, self.loss_function],
                                     feed_dict={self.Q: batch_qs, self.T: batch_ts, self.Y: batch_ys,
                                                self.keep_prob: 1.0})
                training_pair_counter += 1
                if i % self.history_sampling_rate == 0:
                    self.cost_history.append(c)
                    test_idx = np.random.choice(np.arange(0, len(X_test)), batch_size)
                    test_q_idx = test_idx[:int(len(test_idx) / 2)]
                    test_t_idx = test_idx[int(len(test_idx) / 2):]
                    test_batch_qs = X_test[test_q_idx]  # Query images
                    test_batch_ts = X_test[test_t_idx]  # Test images
                    test_batch_ys = calculate_score_batch(test_batch_qs, test_batch_ts, self.test_db)
                    test_batch_qs = self.test_db_features[test_q_idx]
                    test_batch_ts = self.test_db_features[test_t_idx]
                    acc = self.sess.run(self.loss_function, feed_dict={self.Q: test_batch_qs, self.T: test_batch_ts,
                                                                       self.Y: test_batch_ys, self.keep_prob: 1.0})
                    self.test_acc_history.append(acc)
                    print("Batch index:", '%04d' % i, "Cost:", c, "Validation accuracy:", acc)

            # Do more precise test after each epoch
            if epoch % self.display_step == 0:
                test_idx = np.random.choice(np.arange(0, len(X_test)), 256)
                test_q_idx = test_idx[:int(len(test_idx) / 2)]
                test_t_idx = test_idx[int(len(test_idx) / 2):]
                test_batch_qs = X_test[test_q_idx]  # Query images
                test_batch_ts = X_test[test_t_idx]  # Test images
                test_batch_ys = calculate_score_batch(test_batch_qs, test_batch_ts, self.test_db)
                test_batch_qs = self.test_db_features[test_q_idx]
                test_batch_ts = self.test_db_features[test_t_idx]
                test_error = self.sess.run(self.loss_function, feed_dict={self.Q: test_batch_qs, self.T: test_batch_ts,
                                                                          self.Y: test_batch_ys, self.keep_prob: 1.0})
                self.test_acc_history.append(test_error)
                print("Epoch:", '%03d' % (epoch + 1), "total trained pairs:", '%09d' % training_pair_counter,
                      "\ttest error =", test_error, "- time used:", time.time() - start_time)
                if save:
                    self.saver.save(self.sess, save_path + self.model_name, global_step=epoch + 1)

        # TODO make final test
        # test_idx = np.random.choice(np.arange(0, len(X_test)), len(X_test))
        # test_batch_qs = X_test[test_idx[:int(len(test_idx) / 2)]]  # Query images
        # test_batch_ts = X_test[test_idx[int(len(test_idx) / 2):]]  # Test images
        # test_batch_ys = calculate_score_batch(test_batch_qs, test_batch_ts, self.test_db)
        # test_batch_qs = self.feature_extractor.run_inference_on_images(test_batch_qs,
        #                                                                path=self.test_db_path + '/pics/*/')
        # test_batch_ts = self.feature_extractor.run_inference_on_images(test_batch_ts,
        #                                                                path=self.test_db_path + '/pics/*/')
        # test_error = self.sess.run(self.loss_function, feed_dict={self.Q: test_batch_qs, self.T: test_batch_ts,
        #                                                           self.Y: test_batch_ys, self.keep_prob: 1.0})
        # self.test_acc_history.append(test_error)
        # print("Epoch:", '%04d' % (epoch + 1), "\ttotal training pairs:", '%07d' % training_pair_counter,
        #       "\ttest error =", test_error)

        # Printing out some comparisons
        test_idx = np.random.choice(np.arange(0, len(X_test)), 200)
        test_q_idx = test_idx[:int(len(test_idx) / 2)]
        test_t_idx = test_idx[int(len(test_idx) / 2):]
        test_batch_qs = X_test[test_q_idx]  # Query images
        test_batch_ts = X_test[test_t_idx]  # Test images
        test_batch_ys = calculate_score_batch(test_batch_qs, test_batch_ts, self.test_db)
        test_batch_qs = self.test_db_features[test_q_idx]
        test_batch_ts = self.test_db_features[test_t_idx]
        output = self.sess.run(self.y_pred,
                               feed_dict={self.Q: test_batch_qs, self.T: test_batch_ts, self.keep_prob: 1.0})
        print(test_batch_ys)
        print(output)

        if show_test_acc:
            y_axis = np.array(self.test_acc_history)
            plt.plot(y_axis)
            plt.show()

        if show_cost:
            y_axis = np.array(self.cost_history)
            plt.plot(y_axis)
            plt.show()

    def query(self, query_img_name, path='validate/pics/*/'):
        if self.db_features is None:
            print("Running feature extracting on db images")
            self.db_features = self.feature_extractor.run_inference_on_images(self.db)
            print("Finished extracting features from db")

        # Find features for query img
        # TODO possible to check if the img is a part of self.db and use the pre-computed features
        db_keys = list(self.db.keys())
        query_features = self.feature_extractor.run_inference_on_image(query_img_name, path=path)
        if query_features is None:
            return None
        multi_query_features = np.tile(query_features, (len(self.db_features), 1))
        equality_scores = self.sess.run(self.y_pred, feed_dict={self.Q: multi_query_features, self.T: self.db_features,
                                                                self.keep_prob: 1.0})
        eq_threshold = equality_scores.mean() + (5 * equality_scores.std())
        similar_imgs = []

        # TODO make this with np.where()
        for i, eq_s in enumerate(equality_scores):
            if eq_s > eq_threshold:
                similar_imgs.append(db_keys[i])

        # TODO predict all equality scores at once! Would possibly be much faster!
        # for i, pred in enumerate(self.db_features):
        #     if pred is not None and db_keys[i] != query_img_name:
        #         equality_score = self.sess.run(self.y_pred, feed_dict={self.Q: [query_features], self.T: [pred],
        #                                                                self.keep_prob: 1.0})
        #         if random.random() < 0.1:
        #             print("EQ score:", equality_score)
        #         if equality_score > 0.1:  # TODO find a way to fine tune this threshold
        #             similar_imgs.append(self.db[i])
        return similar_imgs


if __name__ == '__main__':
    start_time = time.time()

    # Train diffnet
    # net = DiffNet('train', 'validate')
    # net.train(training_epochs=1, learning_rate=0.01, batch_size=64, save=True, show_cost=True, show_test_acc=True)

    test = True
    if test:
        # Test diffnet
        net = DiffNet('validate')
        net.restore('diffnet1_relu/')
        test_labels = generate_dict_from_directory(pickle_file='./validate/pickle/combined.pickle',
                                                   directory='./validate/txt/')
        test_ids = list(test_labels.keys())[:1000]
        scores = []
        for j, query_img in enumerate(test_ids):
            cluster = net.query(query_img)
            if cluster is not None and len(cluster) > 0:
                score_res = score(test_labels, target=query_img, selection=cluster, n=len(cluster))
                scores.append(score_res)
                print('%05d' % j, "\t", query_img, "- cluster size:", '%03d' % len(cluster), "- score", score_res,
                      "- avg:", reduce(lambda x, y: x + y, scores) / len(scores))
            else:
                scores.append(0.0)
                print('%05d' % j, "\t", query_img, "- No similar images found...", "- avg:",
                      reduce(lambda x, y: x + y, scores) / len(scores))
        print("Average over 100:", np.mean(np.array(scores)))
    print("Time used:", time.time() - start_time)
