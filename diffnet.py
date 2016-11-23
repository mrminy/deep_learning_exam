import pickle
import numpy as np
import time

from functools import reduce
from matplotlib import pyplot as plt
import tensorflow as tf
from imgnet_test import ImageFeatureExtractor
from data_fetcher import calculate_score_batch, load_label_list
from front import generate_dict_from_directory, score


class DiffNet:
    def __init__(self, db_path, test_db_path=None, db_features_path='train_db_features.npy',
                 test_db_features_path='test_db_features.npy'):
        self.feature_extractor = ImageFeatureExtractor()  # Inception-v3
        self.db_path = db_path
        self.test_db_path = test_db_path
        self.db = pickle.load(open('./' + db_path + '/pickle/combined.pickle', 'rb'))
        self.test_db = None
        if test_db_path is not None:
            self.test_db = pickle.load(open('./' + test_db_path + '/pickle/combined.pickle', 'rb'))
        self.db_features = None
        self.test_db_features = None
        self.db_features_name = db_features_path
        self.test_db_features_name = test_db_features_path

        self.history_sampling_rate = 100
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
        self.n_input = 1008 * 2  # 2048 features from second last layer in Inception-v3
        self.n_output = 1  # Equality score
        self.n_hidden_1 = 500  # 1st hidden layer
        self.n_hidden_2 = 100  # 2nd hidden layer

        self.model_name = 'diff_net.ckpt'
        self.cost_history = []
        self.test_acc_history = []

    def build(self, learning_rate=0.01):
        print("Building graph...")
        with self.graph.as_default():
            # Encode curr_state, add transition prediction with selected action and decode to predicted output state
            self.Q = tf.placeholder("float", [None, int(self.n_input)])  # query placeholder
            self.T = tf.placeholder("float", [None, int(self.n_input / 2)])  # test placeholder
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

            # layer_1_q = tf.nn.relu(tf.add(tf.matmul(self.Q, weights['h1']), biases['b1']))
            # layer_1_drop_q = tf.nn.dropout(layer_1_q, self.keep_prob)  # Dropout layer
            # layer_1_t = tf.nn.relu(tf.add(tf.matmul(self.T, weights['h1']), biases['b1']))
            # layer_1_drop_t = tf.nn.dropout(layer_1_t, self.keep_prob)  # Dropout layer
            # concat_tensor = tf.concat(1, [layer_1_drop_q, layer_1_drop_t])

            # concat_tensor = tf.concat(1, [self.Q, self.T])
            layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(self.Q, weights['h1']), biases['b1']))
            layer_1_drop = tf.nn.dropout(layer_1, self.keep_prob)  # Dropout layer
            layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1_drop, weights['h2']), biases['b2']))
            layer_2_drop = tf.nn.dropout(layer_2, self.keep_prob)  # Dropout layer
            out = tf.add(tf.matmul(layer_2_drop, weights['out']), biases['bout'])
            # out = tf.nn.softmax(tf.add(tf.matmul(layer_2_drop, weights['out']), biases['bout']))
            # out = tf.nn.dropout(out, self.keep_prob)  # Dropout layer

            # Prediction
            self.y_pred = out

            # Evaluate model
            correct_prediction = tf.equal(tf.argmax(self.Y, 1), tf.argmax(self.y_pred, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

            # Define loss, minimize the squared error (with or without scaling)
            beta = 0.001
            # self.loss_function = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(self.y_pred, self.Y))

            # self.loss_function = tf.reduce_mean(
            #     tf.nn.softmax_cross_entropy_with_logits(self.y_pred, self.Y) + beta * tf.nn.l2_loss(
            #         weights['h1']) + beta * tf.nn.l2_loss(biases['b1']) + beta * tf.nn.l2_loss(
            #         weights['h2']) + beta * tf.nn.l2_loss(biases['b2']) + beta * tf.nn.l2_loss(
            #         weights['out']) + beta * tf.nn.l2_loss(biases['bout']))

            # self.loss_function = tf.reduce_mean(-tf.reduce_sum(
            #     ((self.Y * tf.log(self.y_pred + 1e-9)) + ((1 - self.Y) * tf.log(1 - self.y_pred + 1e-9)))))
            self.loss_function = tf.reduce_mean(tf.square(self.Y - self.y_pred))  # + beta * tf.nn.l2_loss(
            # weights['h1']) + beta * tf.nn.l2_loss(biases['b1']) + beta * tf.nn.l2_loss(
            # weights['h2']) + beta * tf.nn.l2_loss(biases['b2']) + beta * tf.nn.l2_loss(
            # weights['out']) + beta * tf.nn.l2_loss(biases['bout']))
            self.optimizer = tf.train.AdamOptimizer(learning_rate).minimize(self.loss_function)
            # self.optimizer = tf.train.RMSPropOptimizer(learning_rate, momentum=0.01).minimize(self.loss_function)
            # self.optimizer = tf.train.GradientDescentOptimizer(5.).minimize(self.loss_function)

            # Creates a saver
            self.saver = tf.train.Saver()

            # Initializing the variables
            self.init = tf.initialize_all_variables()

            # Launch the graph
            # config = tf.ConfigProto()
            # config.gpu_options.allow_growth = True
            # config = tf.ConfigProto(log_device_placement=True, device_count={'GPU': 0})
            # self.sess = tf.Session(graph=self.graph, config=config)
            self.sess = tf.Session(graph=self.graph)
            self.sess.run(self.init)

    def restore(self, path, global_step=1):
        if self.sess is None:
            self.build()
        self.saver.restore(self.sess, path + self.model_name + '-' + str(global_step))
        print("Restored Diffnet model")

    def load_image_features(self, load_test_features=False):
        if self.db_features is None:
            try:
                # self.db_features = np.load(self.db_features_name)
                with open(self.db_features_name, 'rb') as handle:
                    self.db_features = pickle.load(handle)
            except IOError:
                pass
            if self.db_features is None:
                print("Running feature extracting on db images")
                self.db_features = self.feature_extractor.run_inference_on_images(self.db,
                                                                                  path=self.db_path + '/pics/*/',
                                                                                  save_name=self.db_features_name)
                print("Finished extracting features from db")
                # self.db_features /= max(self.db_features.max(), abs(self.db_features.min()))
        if load_test_features and self.test_db_features is None:
            try:
                # self.test_db_features = np.load(self.test_db_features_name)
                with open(self.test_db_features_name, 'rb') as handle:
                    self.test_db_features = pickle.load(handle)
            except IOError:
                pass
            if self.test_db_features is None:
                print("Running feature extracting on test db images")
                self.test_db_features = self.feature_extractor.run_inference_on_images(self.test_db,
                                                                                       path=self.test_db_path + '/pics/*/',
                                                                                       save_name=self.test_db_features_name)
                print("Finished extracting features from test db")
                #     self.test_db_features /= max(self.test_db_features.max(), abs(self.test_db_features.min()))

    def train(self, training_epochs=20, learning_rate=0.01, batch_size=32, show_cost=False, show_test_acc=False,
              save=False, save_path='diffnet1/', logger=True):
        # Load and preprocess data
        if logger:
            print("Loading and preprocessing data...")
        X_train = load_label_list(self.db)
        X_test = load_label_list(self.test_db)

        if self.sess is None:
            self.build(learning_rate=learning_rate)

        self.load_image_features(load_test_features=True)

        total_batch = int(len(X_train) / batch_size)
        if logger:
            print("Starting training...")
            print("Total nr of batches:", total_batch)
        # Training cycle
        training_pair_counter = 0
        for epoch in range(training_epochs):
            # Loop over all batches
            idexes = np.arange(len(X_train))
            for i in range(total_batch):
                idx = np.random.choice(idexes, (int(batch_size * 2)), replace=True)
                # q_idx = idx[:(len(idx) - int(len(idx) / 3))]
                # t_idx = list(idx[(len(idx) - int(len(idx) / 3)):]) + list(q_idx[int(len(idx) / 3):])
                q_idx = idx[:int(len(idx) / 2)]
                t_idx = idx[int(len(idx) / 2):]
                batch_qs = X_train[q_idx]  # Query images
                batch_ts = X_train[t_idx]  # Test images
                # Shuffling
                p = np.random.permutation(len(q_idx))
                batch_qs = batch_qs[p]
                batch_ts = batch_ts[p]
                batch_qs_f = [self.db_features[x] for x in batch_qs]
                batch_ts_f = [self.db_features[x] for x in batch_ts]
                batch_qs_f, batch_ts_f, batch_ys = calculate_score_batch(batch_qs, batch_ts, self.db,
                                                                         q_features=batch_qs_f, t_features=batch_ts_f)
                # Run optimization op (backprop) and cost op (to get loss value)
                _, c = self.sess.run([self.optimizer, self.loss_function],
                                     feed_dict={self.Q: batch_qs_f, self.T: batch_ts_f, self.Y: batch_ys,
                                                self.keep_prob: .5})

                training_pair_counter += len(batch_qs)
                if i % self.history_sampling_rate == 0:
                    self.cost_history.append(c)
                    test_idx = np.random.choice(np.arange(0, len(X_test)), batch_size)
                    test_q_idx = test_idx[:int(len(test_idx) / 2)]
                    test_t_idx = test_idx[int(len(test_idx) / 2):]
                    test_batch_qs = X_test[test_q_idx]  # Query images
                    test_batch_ts = X_test[test_t_idx]  # Test images
                    test_batch_qs_f = [self.test_db_features[x] for x in test_batch_qs]
                    test_batch_ts_f = [self.test_db_features[x] for x in test_batch_ts]
                    test_batch_qs_f, test_batch_ts_f, test_batch_ys = calculate_score_batch(test_batch_qs,
                                                                                            test_batch_ts, self.test_db,
                                                                                            q_features=test_batch_qs_f,
                                                                                            t_features=test_batch_ts_f)
                    acc = self.sess.run(self.loss_function, feed_dict={self.Q: test_batch_qs_f, self.T: test_batch_ts_f,
                                                                       self.Y: test_batch_ys, self.keep_prob: 1.})
                    self.test_acc_history.append(acc)
                    print("Batch index:", '%04d' % i, "Cost:", c, "Validation accuracy:", acc)

            # Do more precise test after each epoch
            if epoch % self.display_step == 0:
                test_idx = np.random.choice(np.arange(0, len(X_test)), 1000)
                test_q_idx = test_idx[:int(len(test_idx) / 2)]
                test_t_idx = test_idx[int(len(test_idx) / 2):]
                test_batch_qs = X_test[test_q_idx]  # Query images
                test_batch_ts = X_test[test_t_idx]  # Test images
                test_batch_qs_f = [self.test_db_features[x] for x in test_batch_qs]
                test_batch_ts_f = [self.test_db_features[x] for x in test_batch_ts]
                test_batch_qs_f, test_batch_ts_f, test_batch_ys = calculate_score_batch(test_batch_qs, test_batch_ts,
                                                                                        self.test_db,
                                                                                        q_features=test_batch_qs_f,
                                                                                        t_features=test_batch_ts_f)
                test_accuracy = self.sess.run(self.loss_function,
                                              feed_dict={self.Q: test_batch_qs_f, self.T: test_batch_ts_f,
                                                         self.Y: test_batch_ys, self.keep_prob: 1.})
                # self.test_acc_history.append(test_accuracy)
                print("Epoch:", '%03d' % (epoch + 1), "total trained pairs:", '%09d' % training_pair_counter,
                      "\ttest acc =", test_accuracy, "- time used:", time.time() - start_time)
                if save:
                    self.saver.save(self.sess, save_path + self.model_name, global_step=epoch + 1)

        # Printing out some comparisons
        test_idx = np.random.choice(np.arange(0, len(X_test)), 200)
        test_q_idx = test_idx[:int(len(test_idx) / 2)]
        test_t_idx = test_idx[int(len(test_idx) / 2):]
        test_batch_qs = X_test[test_q_idx]  # Query images
        test_batch_ts = X_test[test_t_idx]  # Test images
        test_batch_qs_f = [self.test_db_features[x] for x in test_batch_qs]
        test_batch_ts_f = [self.test_db_features[x] for x in test_batch_ts]
        test_batch_qs_f, test_batch_ts_f, test_batch_ys = calculate_score_batch(test_batch_qs, test_batch_ts,
                                                                                self.test_db,
                                                                                q_features=test_batch_qs_f,
                                                                                t_features=test_batch_ts_f)
        output = self.sess.run(self.y_pred, feed_dict={self.Q: test_batch_qs_f, self.T: test_batch_ts_f,
                                                       self.Y: test_batch_ys, self.keep_prob: 1.})
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

    def load_numpy_features(self, db_features):
        features = []
        for img_name in load_label_list(db_features):
            features.append(db_features[img_name])
        return np.array(features)

    def query(self, query_img_name, path='validate/pics/*/'):
        self.load_image_features()

        numpy_features = self.load_numpy_features(self.db_features)

        # Find features for query img
        # TODO possible to check if the img is a part of self.db and use the pre-computed features
        db_keys = load_label_list(self.db)
        query_features = self.feature_extractor.run_inference_on_image(query_img_name, path=path)
        if query_features is None:
            return None
        multi_query_features = np.tile(query_features, (len(numpy_features), 1))
        multi_query_features = np.concatenate((multi_query_features, numpy_features), axis=1)
        equality_scores = self.sess.run(self.y_pred, feed_dict={self.Q: multi_query_features, self.keep_prob: 1.0})
        eq_threshold = equality_scores.max() - equality_scores.std()
        print("MAX:", equality_scores.max(), "MIN:", equality_scores.min(), "MEAN:", equality_scores.mean(), "STD:",
              equality_scores.std())
        similar_imgs = []

        # TODO make this with np.where()
        for i, eq_s in enumerate(equality_scores):
            if eq_s[0] > eq_threshold:
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

    # train_features_name = '2048_features/train_db_features.pickle'
    # test_features_name = '2048_features/test_db_features.pickle'
    train_features_name = '1008_features/train_db_features.pickle'
    test_features_name = '1008_features/test_db_features.pickle'

    # Train diffnet
    # net = DiffNet('train', 'validate', db_features_path=train_features_name, test_db_features_path=test_features_name)
    # net.train(training_epochs=100, learning_rate=.001, batch_size=64, save=True, show_cost=False, show_test_acc=False)

    test = True
    if test:
        # Test diffnet
        net = DiffNet('validate', db_features_path=test_features_name)
        net.restore('diffnet1/', global_step=100)
        test_labels = generate_dict_from_directory(pickle_file='./validate/pickle/combined.pickle',
                                                   directory='./validate/txt/')
        test_ids = list(test_labels.keys())[:1000]
        scores = []
        for j, query_img in enumerate(test_ids):
            cluster = net.query(query_img)
            if cluster is not None and len(cluster) > 0:
                score_res = score(test_labels, target=query_img, selection=cluster)
                scores.append(score_res)
                print('%05d' % j, "\t", query_img, "- cluster size:", '%03d' % len(cluster), "- score", score_res,
                      "- avg:", reduce(lambda x, y: x + y, scores) / len(scores))
            else:
                scores.append(0.0)
                print('%05d' % j, "\t", query_img, "- No similar images found...", "- avg:",
                      reduce(lambda x, y: x + y, scores) / len(scores))
        print("Average over 100:", np.mean(np.array(scores)))
    print("Time used:", time.time() - start_time)
