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


def make_one_hot_labels():
    nr_of_labels, all_lables = check_nr_of_labels()
    print("Located all labels")
    train_labels = generate_dict_from_directory()
    labels = []
    counter = 0
    for i in train_labels:
        if counter > 2000:
            break
        if counter > 1000:
            if counter % 1000 == 0:
                print("Image:", i, "iteration:", counter)
            one_hot = np.zeros(nr_of_labels)
            for lable in train_labels[i]:
                one_hot[all_lables.index(lable)] = 1.
            labels.append(one_hot)
        counter += 1
    labels = np.array(labels)
    np.save("test_one_hot.npy", np.array(labels))


def load_image(img_name):
    arr = []
    for filename in glob.glob('train/pics/*/' + img_name + '.jpg'):
        arr.append(Image.open(filename))
    if len(arr) == 1:
        return np.array(arr[0]).flatten()
    if len(arr) > 1:
        print("longer than 1", len(arr), arr[0].fp.name)
        return np.array(arr[0]).flatten()
    return None


def make_images():
    # TODO try to restore the array from before
    train_labels = generate_dict_from_directory()
    images = []
    counter = 0
    for i in train_labels:
        if counter > 2000:
            break
        if counter > 1000:
            if counter % 100 == 0:
                print("Image:", i, "iteration:", counter)
            img = load_image(i)
            images.append(img)
        counter += 1
    images = np.array(images)
    np.save('test_images.npy', np.array(images))
    return images


def load_one_hot_labels(path='test_one_hot.npy'):
    return np.load(path)


def load_images(path='test_images.npy'):
    return np.load(path)


def load_data(data_path='learn_images.npy', labels_path='learn_one_hot.npy'):
    return load_images(data_path), load_one_hot_labels(labels_path)


if __name__ == '__main__':
    # make_one_hot_labels()
    make_images()
    # labels = load_one_hot()
    # print("s")
    # data, labels = load_data()
    # print("test")
