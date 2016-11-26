import csv
import pickle
import random

import numpy as np
import tensorflow as tf
import time
from PIL import Image
from matplotlib import pyplot as plt
from os.path import isfile

from sklearn.cluster import FeatureAgglomeration
from sklearn.manifold import TSNE

import inception
from data_fetcher import load_label_list, generate_training_set_one_hots_numpy, find_img_path, \
    generate_training_set_one_hot_indexes
from one_hot_full_trainingset import DiffNet


class FeaturesAutoencoder:
    """
    Neural network that takes image features as input and outputs a vector where each number represents the similarity
    of that image for that index in the db.
    Query image --> Inception-v3 --> Diffnet --> Similar images
    """

    def __init__(self, db, db_path='./train/pics/*/', db_features_path='./1008_features/train_db_features.pickle'):
        self.feature_extractor = None
        if db is None:
            self.db = pickle.load(open('./train/pickle/combined.pickle', 'rb'))
        else:
            self.db = db
        self.db_features = None
        self.db_path = db_path
        self.db_features_name = db_features_path

        self.one_hot_indexes_path = './old_features/old_one_hot_indexes/training_one_hot_indexes-'

        self.all_labels = load_label_list(self.db)

        self.history_sampling_rate = 50
        self.display_step = 1

        self.graph = tf.Graph()

        # tf placeholders
        self.Q = None  # Feature input
        self.Y = None  # Feature output
        self.y_pred = None
        self.sess = None
        self.keep_prob = None  # for dropout
        self.saver = None
        self.compressed = None

        # Network Parameters
        self.n_input = len(load_label_list(self.db))  # 1008 features from last layer in Inception-v3
        self.n_output = self.n_input  # Equality score for each image in db
        self.n_hidden_1 = 500  # encoder/decoder hidden layer
        self.n_compressed = 50  # Compressed layer

        self.model_name = 'feature_auto_encoder.ckpt'
        self.cost_history = []
        self.test_acc_history = []

    def build(self, learning_rate=0.01):
        """
        Builds the model
        :param learning_rate: fixed learning rate for the optimizer
        :return: None
        """
        print("Building graph...")
        with self.graph.as_default():
            self.Q = tf.placeholder("float", [None, int(self.n_input)])  # query/input placeholder
            self.Y = tf.placeholder("float", [None, self.n_output])  # similarity prediction
            self.keep_prob = tf.placeholder(tf.float32)  # For dropout

            weights = {
                'encoder_h1': tf.Variable(
                    tf.random_normal([self.n_input, self.n_hidden_1])),
                'compressed': tf.Variable(
                    tf.random_normal([self.n_hidden_1, self.n_compressed])),
                'decoder_h1': tf.Variable(
                    tf.random_normal([self.n_compressed, self.n_hidden_1])),
                'out': tf.Variable(
                    tf.random_normal([self.n_hidden_1, self.n_output])),
            }
            biases = {
                'encoder_b1': tf.Variable(tf.random_normal([self.n_hidden_1])),
                'compressed_b': tf.Variable(tf.random_normal([self.n_compressed])),
                'decoder_b1': tf.Variable(tf.random_normal([self.n_hidden_1])),
                'bout': tf.Variable(tf.random_normal([self.n_output])),
            }

            # layer_1_q = tf.nn.relu(tf.add(tf.matmul(self.Q, weights['h1']), biases['b1']))
            # layer_1_drop_q = tf.nn.dropout(layer_1_q, self.keep_prob)  # Dropout layer
            # layer_1_t = tf.nn.relu(tf.add(tf.matmul(self.T, weights['h1']), biases['b1']))
            # layer_1_drop_t = tf.nn.dropout(layer_1_t, self.keep_prob)  # Dropout layer
            # concat_tensor = tf.concat(1, [layer_1_drop_q, layer_1_drop_t])

            layer_1 = tf.nn.tanh(tf.add(tf.matmul(self.Q, weights['encoder_h1']), biases['encoder_b1']))
            layer_1_drop = tf.nn.dropout(layer_1, self.keep_prob)  # Dropout layer
            self.compressed = tf.add(tf.matmul(layer_1_drop, weights['compressed']), biases['compressed_b'])
            compressed_drop = tf.nn.dropout(tf.nn.tanh(self.compressed), self.keep_prob)  # Dropout layer
            layer_2 = tf.nn.tanh(tf.add(tf.matmul(compressed_drop, weights['decoder_h1']), biases['decoder_b1']))
            layer_2_drop = tf.nn.dropout(layer_2, self.keep_prob)  # Dropout layer
            out = tf.nn.tanh(tf.add(tf.matmul(layer_2_drop, weights['out']), biases['bout']))

            # Prediction
            self.y_pred = out

            # Evaluate model
            # correct_prediction = tf.equal(tf.argmax(self.Y, 1), tf.argmax(self.y_pred, 1))
            # self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

            # Define loss, minimize the squared error (with or without scaling)
            # beta = 0.001
            # self.loss_function = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(self.y_pred, self.Y))

            # self.loss_function = tf.reduce_mean(
            #     tf.nn.softmax_cross_entropy_with_logits(self.y_pred, self.Y) + beta * tf.nn.l2_loss(
            #         weights['out']) + beta * tf.nn.l2_loss(biases['bout']))

            # self.loss_function = tf.reduce_mean(-tf.reduce_sum(
            #     ((self.Y * tf.log(self.y_pred + 1e-9)) + ((1 - self.Y) * tf.log(1 - self.y_pred + 1e-9)))))

            self.loss_function = tf.reduce_mean(tf.square(self.Y - self.y_pred))  # + beta * tf.nn.l2_loss(
            # weights['out']) + beta * tf.nn.l2_loss(biases['bout']))

            # Optimizers
            self.optimizer = tf.train.AdamOptimizer(learning_rate).minimize(self.loss_function)
            # self.optimizer = tf.train.RMSPropOptimizer(learning_rate, momentum=0.7).minimize(self.loss_function)
            # self.optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(self.loss_function)

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
        """
        Restores a pre-trained model
        :param path: path for trained model
        :param global_step: to select a specific checkpoint
        :return: None
        """
        if self.sess is None:
            self.build()
        if global_step is None:
            self.saver.restore(self.sess, path + self.model_name)
        else:
            self.saver.restore(self.sess, path + self.model_name + '-' + str(global_step))
        print("Restored model")

    def load_image_features(self):
        """
        Loads/generates image features from Inception-v3
        :return: None
        """
        if self.db_features is None:
            print("Loading image features...")
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

    # def load_one_hots(self, nr):
    #     try:
    #         return pickle.load(open('./training_one_hot_indexes-' + str(nr) + '.pickle', 'rb'))
    #     except IOError:
    #         # Generating one_hots
    #         print("Could not find any one hot labels. Generating...")
    #         generate_training_set_one_hot_indexes(self.db)
    #         return pickle.load(open('./training_one_hot_indexes-' + str(nr) + '.pickle', 'rb'))

    def train(self, training_epochs=20, learning_rate=0.01, batch_size=32, show_cost=False, show_example=False,
              save=False, save_path='diffnet3/', logger=True):
        """
        Trains the network
        :param training_epochs: number of epochs to train for. One epoch is one iteration through the training set
        :param learning_rate: the fixed learning rate
        :param batch_size: training batch size
        :param show_cost: show plot for the cost history after training
        :param show_example: show an example at the end of training
        :param save: if the model should be saved or not
        :param save_path: path for where the model is saved
        :param logger: show unnecessary prints during training
        :return: None
        """
        if self.sess is None:
            self.build(learning_rate=learning_rate)

        self.load_image_features()

        indexes_exists = isfile(self.one_hot_indexes_path + str(0) + '.pickle')
        if not indexes_exists:
            print("Could not find any one-hot index files. Generating them...")
            generate_training_set_one_hot_indexes(self.db)

        training_samples = 0

        if logger:
            print("Starting training...")

        # y_data = None
        for epoch in range(training_epochs):
            # Iterating the saved one_hot dicts
            for d in range(0, 11):
                with open(self.one_hot_indexes_path + str(d) + '.pickle', 'rb') as f:
                    y_data = pickle.load(f)
                    # y_data = self.load_one_hots(d)
                    print("Epoch:", epoch + 1, "- Loading:", d)
                    y_data_labels = np.array(list(y_data.keys()))
                    idexes = np.arange(len(y_data_labels))
                    total_batch = int(len(y_data_labels) / batch_size)

                    # Loop over randomly selected batches
                    for i in range(total_batch):
                        idx = np.random.choice(idexes, batch_size, replace=True)
                        batch_qs = y_data_labels[idx]
                        # batch_qs_f = np.array([self.db_features[x] for x in batch_qs])
                        one_hots = generate_training_set_one_hots_numpy(batch_qs, len(self.all_labels), y_data)

                        # Run optimization and loss function to get loss value
                        _, c = self.sess.run([self.optimizer, self.loss_function],
                                             feed_dict={self.Q: one_hots, self.Y: one_hots, self.keep_prob: 1.})

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
        with open(self.one_hot_indexes_path + str(random.randint(0, 10)) + '.pickle', 'rb') as f:
            y_data = pickle.load(f)
            y_data_labels = np.array(list(y_data.keys()))
            idexes = np.arange(len(y_data_labels))
            idx = np.random.choice(idexes, 1, replace=True)
            batch_qs = y_data_labels[idx]
            # batch_qs_f = np.array([self.db_features[x] for x in batch_qs])
            one_hots = generate_training_set_one_hots_numpy(batch_qs, len(self.all_labels), y_data)
            output = self.sess.run(self.compressed,
                                   feed_dict={self.Q: one_hots, self.keep_prob: 1.})
            print(output)

        # if show_example:
        #     answer = self.all_labels[np.where(batch_qs_f[0] == 1)]
        #     print("Query:", batch_qs[0])
        #     # Select top 6 images
        #     selected = self.all_labels[output[0].argsort()[-6:][::-1]]
        #     equal = 0
        #     for s in selected:
        #         if s in answer:
        #             equal += 1
        #     print("Acc:", equal / len(selected))
        #     print("Len:", len(answer), "Whole:", answer)
        #     print(selected)
        #     for found in selected:
        #         img_path = find_img_path('./train/pics/*/', found)
        #         image = Image.open(img_path)
        #         image.show()

        if show_cost:
            y_axis = np.array(self.cost_history)
            plt.plot(y_axis)
            plt.show()

    def cluster(self, cluster_images, cluster_db_path, save_csv=False):
        compressions = []

        # Finding features
        diffnet = DiffNet(self.db, db_path=self.db_path)
        diffnet.restore('diffnet500/', global_step=10)
        print("Calculating features for", len(cluster_images), "images")
        for img in cluster_images:
            print("Finding features for:", img)
            one_hot = diffnet.feedforward(img, cluster_db_path)
            output = self.sess.run(self.compressed, feed_dict={self.Q: [one_hot], self.keep_prob: 1.})
            compressions.append(output[0])

        # Clustering
        print("Performing clustering...")
        compressions = np.array(compressions)
        # tsne = TSNE(n_components=1, learning_rate=100)
        # X_tsne = tsne.fit_transform(compressions)
        fa = FeatureAgglomeration(n_clusters=10)
        X_clusters = fa.fit_transform(compressions)

        print("Collecting data...")
        csv_dict_arr = []
        for i, img in enumerate(cluster_images):
            csv_dict_arr.append({'1.img': img, '2.class': np.argmax(X_clusters[i]), '3.features': compressions[i]})

        # Saving
        if save_csv:
            keys = load_label_list(csv_dict_arr[0])
            with open('cluster_result.csv', 'w') as output_file:
                dict_writer = csv.DictWriter(output_file, keys, delimiter=';')
                dict_writer.writeheader()
                dict_writer.writerows(csv_dict_arr)

        return csv_dict_arr


if __name__ == '__main__':
    start_time = time.time()
    train_features_path = '1008_features/train_db_features.pickle'

    training_labels = pickle.load(open('./train/pickle/combined.pickle', 'rb'))
    # training_labels = list(training_labels.keys())

    # Training
    net = FeaturesAutoencoder(training_labels, db_path='./train/pics/*/', db_features_path=train_features_path)
    # net.train(training_epochs=20, learning_rate=.003, batch_size=64, save=True, show_cost=True, show_example=False,
    #           save_path='feature_autoencoder/')
    net.restore('./feature_autoencoder/', global_step=9)

    testing_labels = pickle.load(open('./validate/pickle/combined.pickle', 'rb'))
    cluster_lbs = load_label_list(testing_labels)[:1000]
    cluster = net.cluster(cluster_lbs, './validate/pics/*/', save_csv=True)

    print("Time used:", time.time() - start_time)
