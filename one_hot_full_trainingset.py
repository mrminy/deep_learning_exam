import pickle
import random

import numpy as np
import tensorflow as tf
from PIL import Image
from matplotlib import pyplot as plt
from os.path import isfile

import inception
from data_fetcher import load_label_list, generate_training_set_one_hots_numpy, find_img_path, \
    generate_training_set_one_hot_indexes


class DiffNet:
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
        self.n_output = len(self.all_labels)  # Equality score for each image in db
        self.n_hidden_1 = 500  # 1st hidden layer

        self.model_name = 'diff_net.ckpt'
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
                'h1': tf.Variable(
                    tf.random_normal([self.n_input, self.n_hidden_1])),
                'out': tf.Variable(
                    tf.random_normal([self.n_hidden_1, self.n_output])),
            }
            biases = {
                'b1': tf.Variable(tf.random_normal([self.n_hidden_1])),
                'bout': tf.Variable(tf.random_normal([self.n_output])),
            }

            layer_1 = tf.nn.tanh(tf.add(tf.matmul(self.Q, weights['h1']), biases['b1']))
            layer_1_drop = tf.nn.dropout(layer_1, self.keep_prob)  # Dropout layer
            out = tf.nn.tanh(tf.add(tf.matmul(layer_1_drop, weights['out']), biases['bout']))

            # Prediction
            self.y_pred = out

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
                if self.feature_extractor is None:
                    print("Loading Inception-v3")
                    self.feature_extractor = inception.ImageFeatureExtractor()  # Inception-v3
                print("Running feature extracting on db images")
                self.db_features = self.feature_extractor.run_inference_on_images(self.db, path=self.db_path,
                                                                                  save_name=self.db_features_name)
                print("Finished extracting features from db")

    def train(self, training_epochs=10, learning_rate=0.003, batch_size=64, show_cost=False, show_example=False,
              save=False, save_path='model_checkpoints/', logger=True, small_training_set=False):
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

        one_hot_indexes_path = './precomputed_labels/015_threshold_one_hot_indexes-'

        indexes_exists = isfile(one_hot_indexes_path + str(0) + '.pickle')
        if not indexes_exists:
            print("Could not find any one-hot index files. Generating... (this may take a while (10h))")
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
                with open(one_hot_indexes_path + str(d) + '.pickle', 'rb') as f:
                    y_data = pickle.load(f)
                    print("Epoch:", epoch + 1, "- Loading:", d)
                    y_data_labels = np.array(list(y_data.keys()))
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
                                             feed_dict={self.Q: batch_qs_f, self.Y: one_hots, self.keep_prob: .8})

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
            with open(one_hot_indexes_path + str(random.randint(0, max_one_hot_files-1)) + '.pickle', 'rb') as f:
                y_data = pickle.load(f)
                y_data_labels = np.array(list(y_data.keys()))
                idexes = np.arange(len(y_data_labels))
                idx = np.random.choice(idexes, 1, replace=True)
                batch_qs = y_data_labels[idx]
                batch_qs_f = np.array([self.db_features[x] for x in batch_qs])
                one_hots = generate_training_set_one_hots_numpy(batch_qs, len(self.all_labels), y_data)
                output = self.sess.run(self.y_pred,
                                       feed_dict={self.Q: batch_qs_f, self.keep_prob: 1.})

            answer = self.all_labels[np.where(one_hots[0] == 1)]
            print("Query:", batch_qs[0])
            # Select top 6 images
            selected = self.all_labels[output[0].argsort()[-6:][::-1]]
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
            # Showing plot for loss function
            y_axis = np.array(self.cost_history)
            plt.plot(y_axis)
            plt.show()

    def close(self):
        if self.sess is not None:
            self.sess.close()
        self.sess = None

    def feedforward(self, query_img_name, path='./validate/pics/*/'):
        if self.feature_extractor is None:
            print("Loading Inception-v3")
            self.feature_extractor = inception.ImageFeatureExtractor()  # Inception-v3

        # Find features for query img
        query_features = self.feature_extractor.run_inference_on_image(query_img_name, path=path)
        if query_features is None:
            print("Could not extract features from image", query_img_name)
            return []
        output = self.sess.run(self.y_pred, feed_dict={self.Q: [query_features], self.keep_prob: 1.})
        # Selecting images from db that scored above a threshold
        eq_threshold = output[0].max() - 2 * output[0].std()
        one_hot = np.zeros(len(output[0]))
        one_hot[np.where(output[0] >= eq_threshold)[0]] = 1
        return one_hot

    def query(self, query_img_name, path='./validate/pics/*/', show_n_similar_imgs=0):
        """
        Insert a query image and get a cluster of similar images in return
        :param query_img_name: image name
        :param path: folder path to the image
        :param show_n_similar_imgs: number of examples to show for the query
        :return: an array of similar images
        """
        if self.feature_extractor is None:
            print("Loading Inception-v3")
            self.feature_extractor = inception.ImageFeatureExtractor()  # Inception-v3

        # Find features for query img
        query_features = self.feature_extractor.run_inference_on_image(query_img_name, path=path)
        if query_features is None:
            print("Could not extract features from image", query_img_name)
            return []
        output = self.sess.run(self.y_pred, feed_dict={self.Q: [query_features], self.keep_prob: 1.})

        # Showing similar images
        if show_n_similar_imgs > 0:
            np_all_labels = np.array(self.all_labels)
            print("Query:", query_img_name)
            img_path = find_img_path(path, query_img_name)
            image = Image.open(img_path)
            image.show()
            selected = np_all_labels[output[0].argsort()[-show_n_similar_imgs:][::-1]]
            print("Top", show_n_similar_imgs, "similar images:", selected)
            for found in selected:
                img_path = find_img_path(self.db_path, found)
                image = Image.open(img_path)
                image.show()

        # Selecting which images that is similar enough
        eq_threshold = output[0].max() - 2 * output[0].std()
        similar_imgs = self.all_labels[np.where(output[0] >= eq_threshold)[0]]
        if similar_imgs is None:
            similar_imgs = []
        return similar_imgs
