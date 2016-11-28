import glob
import pickle
import time

import numpy as np


def load_label_list(db):
    return np.sort(np.array(list(db.keys())))


def find_img_path(partial_path, img_name):
    for filename in glob.glob(partial_path + img_name + '.jpg'):
        return filename
    return None


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


def generate_training_set_one_hot_indexes(db, score_threshold=0.15):
    one_hots_indexes = {}
    save_counter = 0
    labels = load_label_list(db)
    for j, img_name in enumerate(labels):
        if (j + 1) % 1000 == 0:
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
    print("Generating one-hot labels... ")
    training_labels = pickle.load(open('./train/pickle/combined.pickle', 'rb'))
    generate_training_set_one_hot_indexes(training_labels)
    print("done saving ", time.time() - start_time)
