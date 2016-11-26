import random

import time
from resizeimage import resizeimage

# from front import generate_dict_from_directory
import numpy as np
import pickle
from PIL import Image
import glob


# def check_nr_of_labels():
#     train_labels = generate_dict_from_directory(
#         pickle_file='/home/mikkel/deep_learning_exam/train/pickle/combined.pickle',
#         directory='/home/mikkel/deep_learning_exam/train/txt')
#
#     all_labels = []
#     for i in load_label_list(train_labels):
#         for label in train_labels[i]:
#             if label[0] not in all_labels:
#                 all_labels.append(label[0])
#
#     print(len(all_labels))
#     return len(all_labels), all_labels
#
#
# def make_one_hot_labels(train=True, save_file='train_one_hot.npy'):
#     nr_of_labels, all_lables = check_nr_of_labels()
#     print("Located all labels")
#     if not train:
#         train_labels = generate_dict_from_directory(pickle_file='./validate/pickle/combined.pickle',
#                                                     directory='./validate/txt')
#     else:
#         train_labels = generate_dict_from_directory()
#
#     labels = []
#     counter = 0
#     for i in train_labels:
#         if counter >= 0:
#             if counter % 1000 == 0:
#                 print("Image:", i, "iteration:", counter)
#             one_hot_indexes = []
#             for label in train_labels[i]:
#                 if label in all_lables:
#                     one_hot_indexes.append(all_lables.index(label))
#             labels.append(one_hot_indexes)
#         counter += 1
#     labels = np.array(labels)
#     np.save(save_file, labels)


def load_label_list(db):
    return np.sort(np.array(list(db.keys())))


# def convert_to_one_hot(db):
#     # nr_of_labels = 16825  # Nr of different labels
#
#     print("Generating labels")
#     nr_of_labels, word_array = check_nr_of_labels()
#
#     print("Generating one hots...")
#     one_hots = {}
#     for img_name in load_label_list(db):
#         one_hot = np.zeros(nr_of_labels)
#         label_dict = dict(db[img_name])
#         for label in load_label_list(label_dict):
#             if label in word_array:
#                 index = word_array.index(label)
#                 one_hot[index] = 1.
#         one_hots[img_name] = one_hot
#     return one_hots


def find_img_path(partial_path, img_name):
    for filename in glob.glob(partial_path + img_name + '.jpg'):
        return filename
    return None


def load_image(img_name, train=True, path=None):
    arr = []
    if path is None:
        path = 'train/pics/*/'
        if not train:
            path = 'validate/pics/*/'
    for filename in glob.glob(path + img_name + '.jpg'):
        arr.append(Image.open(filename))
    if len(arr) >= 1:
        img = pre_process_image(arr[0].copy())
        return np.true_divide(np.array(img).flatten(), 255)
    return None


def pre_process_image(img):
    return resizeimage.resize_cover(img, [64, 48])


# def make_images(train=True, save_file='train_images_small.npy'):
#     if not train:
#         train_labels = generate_dict_from_directory(pickle_file='./validate/pickle/combined.pickle',
#                                                     directory='./validate/txt')
#     else:
#         train_labels = generate_dict_from_directory()
#     images = []
#     counter = 0
#     for i in train_labels:
#         if counter % 100 == 0:
#             print("Image:", i, "iteration:", counter)
#         img = load_image(i, train=train)
#         images.append(img)
#         counter += 1
#     images = np.array(images)
#     np.save(save_file, images)
#     return images


# def load_one_hot_labels(path='learn_one_hot.pickle', db=None):
#     try:
#         # self.db_features = np.load(self.db_features_name)
#         with open(path, 'rb') as handle:
#             return pickle.load(handle)
#     except IOError:
#         pass
#     one_hots = convert_to_one_hot(db)
#     with open('/home/mikkel/deep_learning_exam/' + path, 'wb') as handle:
#         pickle.dump(one_hots, handle)
#     return one_hots


def load_images(path='test_images.npy'):
    return np.load(path)


def load_img_feature_vector(path='2048_features/train_db_features.pickle'):
    with open('/home/mikkel/deep_learning_exam/' + path, 'rb') as handle:
        return pickle.load(handle)


# def load_data(numpy=False, data_path='2048_features/train_db_features.pickle', labels_path='learn_one_hot.npy'):
#     feature_vect_dict = load_img_feature_vector(data_path)
#     db = generate_dict_from_directory(
#         pickle_file='/home/mikkel/deep_learning_exam/train/pickle/combined.pickle',
#         directory='/home/mikkel/deep_learning_exam/train/txt')
#     one_hots = load_one_hot_labels(labels_path, db)
#     if numpy:
#         feature_numpy = np.array(feature_vect_dict.values(), dtype=float)
#         one_hot_numpy = np.array(one_hots.values(), dtype=float)
#         return feature_numpy, one_hot_numpy
#     return feature_vect_dict, one_hots


# def generate_pair_similarity_index(db):
#     score_db = {}
#     for i, query in enumerate(db):
#         if i % 1000 == 0:
#             print(i, "- Generating for:", query)
#         scores = []
#         Find score
# if query in db.keys():
#     target_dict = dict(db[query])
#     for test in db:
#         if query != test:
#             scores.append((test, calculate_score(target_dict, test, db)))
#     score_db.append(np.array(scores))
# else:
#     print("Could not find any dict!")
#     score_db.append(None)
# score_db = np.array(score_db)
# print(len(score_db))
# np.save('train_score_db.npy', score_db)

def calculate_score_batch(query_batch, test_batch, db, one_hot=False, score_threshold=0.05, q_features=None,
                          t_features=None):
    q_batch = []
    t_batch = []
    ones = 0
    zeros = 0
    diff_threshold = 2
    y = []
    for i, q in enumerate(query_batch):
        # score = calculate_score(q, test_batch[i], db)
        score = np.correlate(q_features[i], t_features[i])
        if one_hot:
            if score > score_threshold and ones + 1 - zeros <= diff_threshold:
                score = np.array([0, 1])
                y.append(score)
                ones += 1
                q_batch.append(query_batch[i])
                t_batch.append(test_batch[i])
            elif zeros + 1 - ones <= diff_threshold:
                score = np.array([1, 0])
                y.append(score)
                zeros += 1
                q_batch.append(query_batch[i])
                t_batch.append(test_batch[i])
        else:
            # score = np.array([1-score, score])
            # if score == 0.0 and random.random() > 0.8:
            #     score = random.random() / 2
            score = [min(1.0, score[0] * 50)]
            y.append(score)
    q_features = np.array(q_features)
    t_features = np.array(t_features)
    q_features /= q_features.max()
    t_features /= t_features.max()
    q_features = np.concatenate((q_features, t_features), axis=1)
    return q_features / q_features.max(), t_features / t_features.max(), np.array(y)


def calculate_score(query, test, db):
    if query in db.keys():
        target_dict = dict(db[query])
    else:
        print("Could not find query dict for", query)
        return 0.0
    best_score = sum(target_dict.values())

    # Avoid problems with div zero. If best_score is 0.0 we will
    # get 0.0 anyway, then best_score makes no difference
    if best_score == 0.0:
        best_score = 1.0

    if test in db.keys():
        selected_dict = dict(db[test])
    else:
        print("Couldn't find " + test + " in the dict keys.")
        return 0.0

    # Extract the shared elements
    common_elements = list(set(selected_dict.keys()) &
                           set(target_dict.keys()))
    if len(common_elements) > 0:
        # for each shared element, the potential score is the
        # level of certainty in the element for each of the
        # images, multiplied together
        element_scores = [selected_dict[element] * target_dict[element] for element in common_elements]
        # We sum the contributions, and that's it
        return sum(element_scores) / best_score
    return 0.0


def generate_word_base(db):
    save_name = './all_labels.pickle'
    labels = []
    try:
        labels = pickle.load(open(save_name, 'rb'))
    except IOError:
        for img_name in load_label_list(db):
            for tpl in db[img_name]:
                lbl = tpl[0]
                if lbl not in labels:
                    labels.append(lbl)
        with open(save_name, 'wb') as handle:
            pickle.dump(labels, handle)
    return labels


def generate_one_hots_dict(db, labels, save=False):
    save_name = './train_one_hot.pickle'
    one_hot_dict = {}
    try:
        one_hot_dict = pickle.load(open(save_name, 'rb'))
    except IOError:
        feature_size = len(labels)

        for img_name in load_label_list(db):
            one_hot = np.zeros(feature_size)
            for tpl in db[img_name]:
                label = tpl[0]
                index = labels.index(label)
                one_hot[index] = 1
            one_hot_dict[img_name] = one_hot
        with open(save_name, 'wb') as handle:
            pickle.dump(one_hot_dict, handle)
    return one_hot_dict


def generate_one_hots_numpy(label_list, all_labels, db):
    one_hots = []
    feature_size = len(all_labels)
    for img_name in label_list:
        one_hot = np.zeros(feature_size)
        for tpl in db[img_name]:
            label = tpl[0]
            if label in all_labels:
                index = all_labels.index(label)
                one_hot[index] = 1
        one_hots.append(one_hot)
    return np.array(one_hots)


def generate_training_set_one_hots_numpy(label_list, feature_size, one_hot_indexes):
    one_hots = []
    one_hot_labels = list(one_hot_indexes.keys())
    for img_name in label_list:
        one_hot = np.zeros(feature_size)
        if img_name in one_hot_labels:
            one_hot[one_hot_indexes[img_name]] = 1
        else:
            print("No", img_name, "in db!")
        one_hots.append(one_hot)
    return np.array(one_hots)


def generate_training_set_one_hots_numpy_simple(label_list, feature_size):
    one_hots = []
    for img_name in label_list:
        one_hot = np.zeros(feature_size)
        one_hot[np.where(label_list == img_name)] = 1
        one_hots.append(one_hot)
    return np.array(one_hots)


def generate_training_set_one_hot_indexes(db, score_threshold=0.15):
    one_hots_indexes = {}
    save_counter = 0
    labels = load_label_list(db)
    for j, img_name in enumerate(labels):
        if (j+1) % 1000 == 0:
            print(j, "of", len(labels), "-", img_name)
        indexes = []
        for i, test_label in enumerate(labels):
            # if img_name != test_label:
            score = calculate_score(img_name, test_label, db)
            if score > score_threshold:
                indexes.append(i)
        one_hots_indexes[img_name] = indexes
        if (j + 1) % 10000 == 0:
            with open('015_threshold_one_hot_indexes-' + str(save_counter) + '.pickle', 'wb') as handle:
                pickle.dump(one_hots_indexes, handle)
            save_counter += 1
            one_hots_indexes = {}
    with open('015_threshold_one_hot_indexes-' + str(save_counter) + '.pickle', 'wb') as handle:
        pickle.dump(one_hots_indexes, handle)


if __name__ == '__main__':
    start_time = time.time()
    training_labels = pickle.load(open('./train/pickle/combined.pickle', 'rb'))
    print(len(load_label_list(training_labels)))
    # generate_training_set_one_hot_indexes(training_labels)
    # all_labels = {}
    # for i in range(0, 10):
    #     training_labels = pickle.load(open('./training_one_hot_indexes-' + str(i) + '.pickle', 'rb'))
    #     all_labels = {**all_labels, **training_labels}

    # labels2 = generate_word_base(training_labels)
    # one_hots = generate_one_hots_dict(training_labels, labels2)

    # generate_pair_similarity_index(training_labels)

    # make_one_hot_labels(train=False, save_file='test_one_hot.npy')
    # make_images(train=False, save_file='test_images_small.npy')
    # path = 'data/small_size/train_images_small.npy'
    # image_set = load_images(path=path)
    # image_set = np.true_divide(image_set, 255)
    # np.save(path, image_set)
    print("done saving zeros", time.time() - start_time)
