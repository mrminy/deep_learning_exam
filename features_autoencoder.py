import argparse
import csv
import pickle
import random
import time
from os.path import isfile

import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from sklearn.cluster import FeatureAgglomeration

from data_fetcher import load_label_list, generate_training_set_one_hots_numpy, generate_training_set_one_hot_indexes
from one_hot_full_trainingset import DiffNet


class FeaturesAutoencoder:
    """
    A deep autoencoder that takes one-hot vectors as input and tries to output the same as the input vector. They key
     is the compression. The middle layer is only 50 nodes, while the input typically is 100k nodes
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
        self.Q = None  # One-hot input
        self.Y = None  # One-hot output
        self.y_pred = None
        self.sess = None
        self.keep_prob = None  # for dropout
        self.saver = None
        self.compressed = None

        # Network Parameters
        self.n_input = len(load_label_list(self.db))  # Typically 100k one-hot index input
        self.n_output = self.n_input  # The same as input
        self.n_hidden_1 = 500  # encoder/decoder size for the hidden layers
        self.n_compressed = 50  # Compressed layer (middle layer)

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
            self.Q = tf.placeholder("float", [None, int(self.n_input)])
            self.Y = tf.placeholder("float", [None, self.n_output])
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

            layer_1 = tf.nn.tanh(tf.add(tf.matmul(self.Q, weights['encoder_h1']), biases['encoder_b1']))
            layer_1_drop = tf.nn.dropout(layer_1, self.keep_prob)  # Dropout layer
            self.compressed = tf.add(tf.matmul(layer_1_drop, weights['compressed']), biases['compressed_b'])
            compressed_drop = tf.nn.dropout(tf.nn.tanh(self.compressed), self.keep_prob)  # Dropout layer
            layer_2 = tf.nn.tanh(tf.add(tf.matmul(compressed_drop, weights['decoder_h1']), biases['decoder_b1']))
            layer_2_drop = tf.nn.dropout(layer_2, self.keep_prob)  # Dropout layer
            out = tf.nn.tanh(tf.add(tf.matmul(layer_2_drop, weights['out']), biases['bout']))

            # Prediction
            self.y_pred = out

            # Mean squared error
            self.loss_function = tf.reduce_mean(tf.square(self.Y - self.y_pred))

            # Optimizer
            self.optimizer = tf.train.AdamOptimizer(learning_rate).minimize(self.loss_function)

            # Creates a saver
            self.saver = tf.train.Saver()

            # Initializing the variables
            self.init = tf.initialize_all_variables()

            # Launch the graph
            config = tf.ConfigProto(log_device_placement=True, device_count={'GPU': 0})
            self.sess = tf.Session(graph=self.graph, config=config)
            self.sess.run(self.init)

    def restore(self, path, global_step=None):
        """
        Restores a pre-trained model
        :param path: path to folder for trained model
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

    def train(self, training_epochs=20, learning_rate=0.01, batch_size=32, show_cost=False, show_example=False,
              save=False, save_path='feature_autoencoder/', logger=True, small_training_set=False):
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

        max_one_hot_files = 10
        if small_training_set:
            max_one_hot_files = 1

        for epoch in range(training_epochs):
            # Iterating the saved one_hot dicts
            for d in range(0, max_one_hot_files):
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

        if show_example:
            # Running one example to check accuracy and similar images
            with open(self.one_hot_indexes_path + str(random.randint(0, max_one_hot_files - 1)) + '.pickle', 'rb') as f:
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

        if show_cost:
            y_axis = np.array(self.cost_history)
            plt.plot(y_axis)
            plt.show()

    def cluster(self, cluster_images, cluster_db_path, diffnet_paht='model_checkpoints/', save_csv=False):
        compressions = []

        # Finding features
        diffnet = DiffNet(self.db, db_path=self.db_path)
        diffnet.restore(diffnet_paht)
        print("Calculating features for", len(cluster_images), "images")
        for img in cluster_images:
            print("Finding features for:", img)
            one_hot = diffnet.feedforward(img, cluster_db_path)
            output = self.sess.run(self.compressed, feed_dict={self.Q: [one_hot], self.keep_prob: 1.})
            compressions.append(output[0])

        # Clustering
        print("Performing clustering...")
        compressions = np.array(compressions)
        fa = FeatureAgglomeration(n_clusters=30)
        X_clusters = fa.fit_transform(compressions)

        print("Collecting data...")
        csv_dict_arr = []
        for i, img in enumerate(cluster_images):
            csv_dict_arr.append({'1.img': img, '2.class': np.argmax(X_clusters[i]), '3.features': compressions[i]})

        # Saving
        if save_csv:
            print("Saving data to csv...")
            keys = load_label_list(csv_dict_arr[0])
            with open('cluster_result.csv', 'w') as output_file:
                dict_writer = csv.DictWriter(output_file, keys, delimiter=';')
                dict_writer.writeheader()
                dict_writer.writerows(csv_dict_arr)

        return csv_dict_arr


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-train', default=False, type=bool, help='If the model should be trained or not', dest='train')
    parser.add_argument('-small_training_set', default=False, type=bool,
                        help='If you do not have all 100k labels, only the first 10k, this must be True!',
                        dest='small_training_set')
    parser.add_argument('-save_csv', default=True, type=bool,
                        help='If the results should be saved to a csv file or not', dest='save_csv')
    parser.add_argument('-test_path', default="validate", help='Path to pickle files that should be clustered',
                        dest='test_path')
    parser.add_argument('-restore_path', default="./feature_autoencoder/", help='Path to saved model',
                        dest='restore_path')

    args = parser.parse_args()

    start_time = time.time()
    train_features_path = '1008_features/train_db_features.pickle'

    training_labels = pickle.load(open('./train/pickle/combined.pickle', 'rb'))

    net = FeaturesAutoencoder(training_labels, db_path='./train/pics/*/', db_features_path=train_features_path)

    if args.train:
        # Training for 20 epochs
        net.train(training_epochs=20, learning_rate=.003, batch_size=64, save=True, show_cost=True, show_example=False,
                  save_path=args.restore_path, small_training_set=args.small_training_set)
    else:
        # Restoring pre-trained model
        net.restore(args.restore_path)

    testing_labels = pickle.load(open('./' + args.test_path + '/pickle/combined.pickle', 'rb'))
    cluster_lbs = load_label_list(testing_labels)
    cluster = net.cluster(cluster_lbs, './' + args.test_path + '/pics/*/', save_csv=args.save_csv)

    print("Time used:", time.time() - start_time)
