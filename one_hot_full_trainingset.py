import pickle
import numpy as np
import time

from functools import reduce

from PIL import Image
from matplotlib import pyplot as plt
import tensorflow as tf
import imgnet_test
from data_fetcher import calculate_score_batch, load_label_list, generate_one_hots_numpy, \
    generate_training_set_one_hots_numpy, generate_training_set_one_hots_numpy_simple, find_img_path
from front import generate_dict_from_directory, score


class DiffNet:
    def __init__(self, db, db_path='./train/pics/*/', db_features_path='./1008_features/train_db_features.pickle'):
        self.feature_extractor = None
        if db is None:
            self.db = pickle.load(open('./train/pickle/combined.pickle', 'rb'))
        else:
            self.db = db
        self.db_features = None
        self.db_path = db_path
        self.db_features_name = db_features_path

        self.all_labels = load_label_list(self.db)

        self.history_sampling_rate = 50
        self.display_step = 1

        self.graph = tf.Graph()

        # tf placeholders
        self.Q = None  # Query input
        self.Y = None  # One hots
        self.y_pred = None
        self.sess = None
        self.keep_prob = None  # for dropout
        self.saver = None

        # Network Parameters
        self.n_input = 1008  # 1008 features from last layer in Inception-v3
        self.n_output = len(load_label_list(self.db))  # Equality score for each image in db
        # self.n_hidden_1 = 1  # 1st hidden layer
        # self.n_hidden_2 = 1  # 2nd hidden layer

        self.model_name = 'diff_net.ckpt'
        self.cost_history = []
        self.test_acc_history = []

    def build(self, learning_rate=0.01):
        print("Building graph...")
        with self.graph.as_default():
            self.Q = tf.placeholder("float", [None, int(self.n_input)])  # query/input placeholder
            self.Y = tf.placeholder("float", [None, self.n_output])  # similarity prediction
            self.keep_prob = tf.placeholder(tf.float32)  # For dropout

            weights = {
                # 'h1': tf.Variable(
                #     tf.random_normal([self.n_input, self.n_hidden_1])),
                # 'h2': tf.Variable(
                #     tf.random_normal([self.n_hidden_1, self.n_hidden_2])),
                'out': tf.Variable(
                    tf.random_normal([self.n_input, self.n_output])),
            }
            biases = {
                # 'b1': tf.Variable(tf.random_normal([self.n_hidden_1])),
                # 'b2': tf.Variable(tf.random_normal([self.n_hidden_2])),
                'bout': tf.Variable(tf.random_normal([self.n_output])),
            }

            # layer_1_q = tf.nn.relu(tf.add(tf.matmul(self.Q, weights['h1']), biases['b1']))
            # layer_1_drop_q = tf.nn.dropout(layer_1_q, self.keep_prob)  # Dropout layer
            # layer_1_t = tf.nn.relu(tf.add(tf.matmul(self.T, weights['h1']), biases['b1']))
            # layer_1_drop_t = tf.nn.dropout(layer_1_t, self.keep_prob)  # Dropout layer
            # concat_tensor = tf.concat(1, [layer_1_drop_q, layer_1_drop_t])

            # concat_tensor = tf.concat(1, [self.Q, self.T])
            # layer_1 = tf.nn.tanh(tf.add(tf.matmul(self.Q, weights['h1']), biases['b1']))
            # layer_1_drop = tf.nn.dropout(layer_1, self.keep_prob)  # Dropout layer
            # layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1_drop, weights['h2']), biases['b2']))
            # layer_2_drop = tf.nn.dropout(layer_2, self.keep_prob)  # Dropout layer
            # out = tf.add(tf.matmul(self.Q, weights['out']), biases['bout'])
            out = tf.nn.tanh(tf.add(tf.matmul(self.Q, weights['out']), biases['bout']))
            # out = tf.nn.dropout(out, self.keep_prob)  # Dropout layer

            # Prediction
            self.y_pred = out

            # Evaluate model
            # correct_prediction = tf.equal(tf.argmax(self.Y, 1), tf.argmax(self.y_pred, 1))
            # self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

            # Define loss, minimize the squared error (with or without scaling)
            beta = 0.001
            # self.loss_function = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(self.y_pred, self.Y))

            # self.loss_function = tf.reduce_mean(
            #     tf.nn.softmax_cross_entropy_with_logits(self.y_pred, self.Y))
            # + beta * tf.nn.l2_loss(
            #         weights['out']) + beta * tf.nn.l2_loss(biases['bout']))

            # self.loss_function = tf.reduce_mean(-tf.reduce_sum(
            #     ((self.Y * tf.log(self.y_pred + 1e-9)) + ((1 - self.Y) * tf.log(1 - self.y_pred + 1e-9)))))
            self.loss_function = tf.reduce_mean(tf.square(self.Y - self.y_pred))  # + beta * tf.nn.l2_loss(
            # weights['h1']) + beta * tf.nn.l2_loss(biases['b1']) + beta * tf.nn.l2_loss(
            # weights['h2']) + beta * tf.nn.l2_loss(biases['b2']) + beta * tf.nn.l2_loss(
            # weights['out']) + beta * tf.nn.l2_loss(biases['bout']))

            # Optimizers
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
            config = tf.ConfigProto(log_device_placement=True, device_count={'GPU': 0})
            # self.sess = tf.Session(graph=self.graph, config=config)
            self.sess = tf.Session(graph=self.graph, config=config)
            self.sess.run(self.init)

    def restore(self, path, global_step=None):
        if self.sess is None:
            self.build()
        if global_step is None:
            self.saver.restore(self.sess, path + self.model_name)
        else:
            self.saver.restore(self.sess, path + self.model_name + '-' + str(global_step))
        print("Restored model")

    def load_image_features(self):
        if self.db_features is None:
            try:
                with open(self.db_features_name, 'rb') as handle:
                    self.db_features = pickle.load(handle)
            except IOError:
                pass
            if self.db_features is None:
                print("Running feature extracting on db images")
                self.db_features = self.feature_extractor.run_inference_on_images(self.db, path=self.db_path,
                                                                                  save_name=self.db_features_name)
                print("Finished extracting features from db")

    def train(self, training_epochs=20, learning_rate=0.01, batch_size=32, show_cost=False, show_example=False,
              save=False, save_path='diffnet3/', logger=True):
        if self.sess is None:
            self.build(learning_rate=learning_rate)

        if logger:
            print("Loading image features...")
        self.load_image_features(load_test_features=True)

        training_samples = 0

        if logger:
            print("Starting training...")

        y_data = None
        for epoch in range(training_epochs):
            # Iterating the saved one_hot dicts
            for d in range(0, 11):
                y_data = pickle.load(open('./training_one_hot_indexes-' + str(d) + '.pickle', 'rb'))
                print("Epoch:", epoch + 1, "- Loading:", d)
                y_data_labels = load_label_list(y_data)
                idexes = np.arange(len(y_data_labels))
                total_batch = int(len(y_data_labels) / batch_size)

                # Loop over randomly selected batches
                for i in range(total_batch):
                    idx = np.random.choice(idexes, batch_size, replace=True)
                    batch_qs = y_data_labels[idx]
                    batch_qs_f = np.array([self.db_features[x] for x in batch_qs])
                    one_hots = generate_training_set_one_hots_numpy(batch_qs, len(self.all_labels), y_data)

                    # Run optimization and loss function to get loss value
                    _, c = self.sess.run([self.optimizer, self.loss_function],
                                         feed_dict={self.Q: batch_qs_f, self.Y: one_hots, self.keep_prob: .7})

                    training_samples += len(batch_qs)
                    if i % self.history_sampling_rate == 0:
                        self.cost_history.append(c)
                        print("Batch:", '%04d' % i, "Cost:", c)

            if epoch % self.display_step == 0:
                print("Epoch:", epoch + 1, "- nr of samples trained:", training_samples)
            if save:
                # Checkpoint after each epoch
                self.saver.save(self.sess, save_path + self.model_name, global_step=epoch + 1)

        if save:
            self.saver.save(self.sess, save_path + self.model_name)

        # Running one example to check accuracy and similar images
        y_data_labels = load_label_list(y_data)
        idexes = np.arange(len(y_data_labels))
        idx = np.random.choice(idexes, 1, replace=True)
        batch_qs = y_data_labels[idx]
        batch_qs_f = np.array([self.db_features[x] for x in batch_qs])
        one_hots = generate_training_set_one_hots_numpy(batch_qs, len(self.all_labels), y_data)
        output = self.sess.run(self.y_pred,
                               feed_dict={self.Q: batch_qs_f, self.keep_prob: 1.})

        if show_example:
            np_all_labels = np.array(self.all_labels)
            answer = np_all_labels[np.where(one_hots[0] == 1)]
            print("Query:", batch_qs[0])
            selected = np_all_labels[output[0].argsort()[-6:][::-1]]
            equal = 0
            for s in selected:
                if s in answer:
                    equal += 1
            print("Acc:", equal / len(selected))
            print("Len:", len(answer), "Whole:", answer)
            print(selected)
            for found in selected:
                img_path = find_img_path('./train/pics/*/', found)
                image = Image.open(img_path)
                image.show()

        if show_cost:
            y_axis = np.array(self.cost_history)
            plt.plot(y_axis)
            plt.show()

    def query(self, query_img_name, path='./validate/pics/*/', show_n_similar_imgs=0):
        if self.feature_extractor is None:
            self.feature_extractor = imgnet_test.ImageFeatureExtractor()  # Inception-v3

        self.load_image_features()

        # Find features for query img
        db_keys = load_label_list(self.db)
        query_features = self.feature_extractor.run_inference_on_image(query_img_name, path=path)
        if query_features is None:
            return None
        output = self.sess.run(self.y_pred, feed_dict={self.Q: [query_features], self.keep_prob: 1.})
        similar_imgs = []
        eq_threshold = 0.01

        # Showing similar images
        if show_n_similar_imgs > 0:
            np_all_labels = np.array(self.all_labels)
            print("Query:", query_img_name)
            selected = np_all_labels[output[0].argsort()[-show_n_similar_imgs:][::-1]]
            print("Top", show_n_similar_imgs, "similar images:", selected)
            for found in selected:
                img_path = find_img_path('./train/pics/*/', found)
                image = Image.open(img_path)
                image.show()

        # TODO make this with np.where()
        for i, o in enumerate(output):
            if o[0] > eq_threshold:
                similar_imgs.append(db_keys[i])
        return similar_imgs


if __name__ == '__main__':
    # Use your_code.py instead!
    pass
    # start_time = time.time()
    #
    # # train_features_name = '2048_features/train_db_features.pickle'
    # # test_features_name = '2048_features/test_db_features.pickle'
    # train_features_name = '1008_features/train_db_features.pickle'
    # test_features_name = '1008_features/test_db_features.pickle'
    #
    # # Train diffnet
    # net = DiffNet()
    # net.train(training_epochs=20, learning_rate=.003, batch_size=64, save=True, show_cost=True, show_test_acc=False,
    #           show_example=True, save_path='diffnet3/')
    #
    # test = False
    # if test:
    #     # Test diffnet
    #     net = DiffNet('validate', db_features_path=test_features_name)
    #     net.restore('diffnet3/', global_step=10)
    #     test_labels = generate_dict_from_directory(pickle_file='./validate/pickle/combined.pickle',
    #                                                directory='./validate/txt/')
    #     test_ids = list(test_labels.keys())[:1000]
    #     scores = []
    #     for j, query_img in enumerate(test_ids):
    #         cluster = net.query(query_img)
    #         if cluster is not None and len(cluster) > 0:
    #             score_res = score(test_labels, target=query_img, selection=cluster)
    #             scores.append(score_res)
    #             print('%05d' % j, "\t", query_img, "- cluster size:", '%03d' % len(cluster), "- score", score_res,
    #                   "- avg:", reduce(lambda x, y: x + y, scores) / len(scores))
    #         else:
    #             scores.append(0.0)
    #             print('%05d' % j, "\t", query_img, "- No similar images found...", "- avg:",
    #                   reduce(lambda x, y: x + y, scores) / len(scores))
    #     print("Average over 100:", np.mean(np.array(scores)))
    # print("Time used:", time.time() - start_time)
