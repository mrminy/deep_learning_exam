import pickle
import random

from functools import reduce

from data_fetcher import load_label_list
from front import generate_dict_from_directory, score
from imgnet_test import ImageFeatureExtractor
import numpy as np


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

    def query(self, query_img_name, path='validate/pics/*/'):
        self.load_image_features()

        # Find features for query img
        # TODO possible to check if the img is a part of self.db and use the pre-computed features
        db_keys = load_label_list(self.db)
        query_features = self.feature_extractor.run_inference_on_image(query_img_name, path=path)
        if query_features is None:
            return None
        # query_features_indexes_max = np.where(query_features > query_features.max() * query_features.std())
        select_max_features = 50
        query_features_indexes_max = query_features.argsort()[-select_max_features:][::-1]
        # query_features_indexes_min = np.squeeze(np.where(query_features == query_features.min()))
        similar_imgs = []
        for t in range(0, select_max_features):
            equality_threshold = select_max_features - t
            for i, test_name in enumerate(load_label_list(self.db_features)):
                if test_name != query_img_name:
                    # test_features_indexes_max = np.where(test_features > test_features.max() * (1-query_features.std()))
                    test_features_indexes_max = self.db_features[test_name].argsort()[-select_max_features:][::-1]
                    # test_features_indexes_min = np.squeeze(np.where(test_features == test_features.min()))
                    nr_of_equal_features_max = np.count_nonzero(query_features_indexes_max == test_features_indexes_max)
                    # nr_of_equal_features_min = np.sum(query_features_indexes_min == test_features_indexes_min)
                    # if nr_of_equal_features_max > 2:
                    #     print(nr_of_equal_features_max)
                    if nr_of_equal_features_max >= equality_threshold:
                        similar_imgs.append(db_keys[i])
                # else:
                #     print("what")

                if len(similar_imgs) >= 30:
                    print("Breaks because of big cluster...")
                    break
            if len(similar_imgs) >= 30:
                break

        return similar_imgs

    def query2(self, query_img_name, path='validate/pics/*/'):
        self.load_image_features()

        # Find features for query img
        # TODO possible to check if the img is a part of self.db and use the pre-computed features
        db_keys = load_label_list(self.db)
        query_features = self.feature_extractor.run_inference_on_image(query_img_name, path=path)
        if query_features is None:
            return None
        similar_imgs = []

        for i, test_name in enumerate(load_label_list(self.db_features)):
            if test_name != query_img_name:
                corr = np.correlate(query_features, self.db_features[test_name])
                # if random.random() < 0.001:
                #     print(corr[0])

                if corr[0] >= 0.2:
                    similar_imgs.append(db_keys[i])

            if len(similar_imgs) >= 40:
                print("Breaks because of big cluster...")
                break

        return similar_imgs


if __name__ == '__main__':
    # test_features_name = '/home/mikkel/deep_learning_exam/2048_features/test_db_features.npy'
    test_features_name = '/home/mikkel/deep_learning_exam/1008_features/test_db_features.pickle'
    net = DiffNet('validate', db_features_path=test_features_name)
    test_labels = generate_dict_from_directory(pickle_file='./validate/pickle/combined.pickle',
                                               directory='./validate/txt/')
    test_ids = load_label_list(test_labels)[:1000]
    scores = []
    for j, query_img in enumerate(test_ids):

        cluster = net.query2(query_img)

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
