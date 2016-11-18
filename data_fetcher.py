from resizeimage import resizeimage

from front import generate_dict_from_directory
import numpy as np
import pickle
from PIL import Image
import glob


def check_nr_of_labels():
    train_labels = generate_dict_from_directory()

    all_labels = []
    for i in train_labels:
        for label in train_labels[i]:
            if label not in all_labels:
                all_labels.append(label)

    print(len(all_labels))
    return len(all_labels), all_labels


def make_one_hot_labels(train=True, save_file='train_one_hot.npy'):
    nr_of_labels, all_lables = check_nr_of_labels()
    print("Located all labels")
    if not train:
        train_labels = generate_dict_from_directory(pickle_file='./validate/pickle/combined.pickle',
                                                    directory='./validate/txt')
    else:
        train_labels = generate_dict_from_directory()

    labels = []
    counter = 0
    for i in train_labels:
        if counter >= 0:
            if counter % 1000 == 0:
                print("Image:", i, "iteration:", counter)
            one_hot_indexes = []
            for label in train_labels[i]:
                if label in all_lables:
                    one_hot_indexes.append(all_lables.index(label))
            labels.append(one_hot_indexes)
        counter += 1
    labels = np.array(labels)
    np.save(save_file, labels)


def convert_to_one_hot(one_hot_indexes):
    nr_of_labels = 16825  # Nr of different labels
    one_hots = []
    for indexes in one_hot_indexes:
        one_hot = np.zeros(nr_of_labels)
        one_hot[indexes] = 1.
        one_hots.append(one_hot)
    return np.array(one_hots)


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


def make_images(train=True, save_file='train_images_small.npy'):
    if not train:
        train_labels = generate_dict_from_directory(pickle_file='./validate/pickle/combined.pickle',
                                                    directory='./validate/txt')
    else:
        train_labels = generate_dict_from_directory()
    images = []
    counter = 0
    for i in train_labels:
        if counter % 100 == 0:
            print("Image:", i, "iteration:", counter)
        img = load_image(i, train=train)
        images.append(img)
        counter += 1
    images = np.array(images)
    np.save(save_file, images)
    return images


def load_one_hot_labels(path='test_one_hot.npy'):
    return np.load(path)


def load_images(path='test_images.npy'):
    return np.load(path)


def load_data(data_path='learn_images.npy', labels_path='learn_one_hot.npy'):
    return load_images(data_path), load_one_hot_labels(labels_path)


def generate_pair_similarity_index(dict):
    score_db = []
    for query in dict:
        print("Generating for:", query)
        scores = []
        for test in dict:
            if query != test:
                # Find score
                score = 0.0
                scores.append((test, score))
        score_db.append(np.array(scores))
    score_db = np.array(score_db)
    np.save('score_db.npy', score_db)


if __name__ == '__main__':
    training_labels = pickle.load(open('./validate/pickle/combined.pickle', 'rb'))
    generate_pair_similarity_index(training_labels)

    # make_one_hot_labels(train=False, save_file='test_one_hot.npy')
    # make_images(train=False, save_file='test_images_small.npy')
    # path = 'data/small_size/train_images_small.npy'
    # image_set = load_images(path=path)
    # image_set = np.true_divide(image_set, 255)
    # np.save(path, image_set)
    print("done saving")
