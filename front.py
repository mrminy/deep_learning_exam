import argparse
import os.path
import random
import glob
import pickle

import time

from your_code import train, test


def score(label_dict, target='', selection=list(), n=50):
    """
    Calculate the score of a selected set compared to the target image.
    :param label_dict: dictionary of labels, keys are image IDs
    :param target: image ID of the query image
    :param selection: the list of IDs retrieved
    :param n: the assumed number of relevant images. Kept fixed at 50
    :return: the calculated score
    """
    # Remove the queried element
    selection = list(set(selection) - {target})

    # k is the number of retrieved elements
    k = len(selection)
    if target in label_dict.keys():
        target_dict = dict(label_dict[target])
    else:
        print("Couldn't find " + target + " in the dict keys.")
        target_dict = {}

    # Current score will accumulate the element-wise scores,
    # before rescaling by scaling by 2/(k*n)
    current_score = 0.0

    # Calculate best possible score of image
    best_score = sum(target_dict.values())

    # Avoid problems with div zero. If best_score is 0.0 we will
    # get 0.0 anyway, then best_score makes no difference
    if best_score == 0.0:
        best_score = 1.0

    # Loop through all the selected elements
    for selected_element in selection:

        # If we have added a non-existing image we will not get
        # anything, and create a dict with no elements
        # Otherwise select the current labels
        if selected_element in label_dict.keys():
            selected_dict = dict(label_dict[selected_element])
        else:
            print("Couldn't find " + selected_element +
                  " in the dict keys.")
            selected_dict = {}

        # Extract the shared elements
        common_elements = list(set(selected_dict.keys()) &
                               set(target_dict.keys()))
        if len(common_elements) > 0:
            # for each shared element, the potential score is the
            # level of certainty in the element for each of the
            # images, multiplied together
            element_scores = [selected_dict[element] *
                              target_dict[element]
                              for element in common_elements]
            # We sum the contributions, and that's it
            current_score += sum(element_scores) / best_score
        else:
            # If there are no shared elements,
            # we won't add anything
            pass

    # We are done after scaling
    return current_score * 2 / (k + n)


def generate_dict_from_directory(pickle_file='./train/pickle/combined.pickle', directory='./train/txt'):
    """
    Goes through a full directory of .txt-files, and transforms the .txt into dict.
    Creates a combo of all the txt-files which is saved in a pickle. Also saves a small .pickle per txt-file.
    :param pickle_file: Where to store the final dict
    :param directory: Directory to traverse for .txt-files
    :return: A dict
    """

    if os.path.isfile(pickle_file):
        # This is easy - dict generated already
        f = open(pickle_file, 'rb')
        my_dict = pickle.load(f)
        f.close()
    else:
        my_dict = {}

        for f in glob.glob(directory + '/*.txt'):
            # Add elements from dict; Requires Python 3.5
            my_dict = {**generate_dict_from_text_file(f), **my_dict}

        f = open(pickle_file, 'wb')
        pickle.dump(my_dict, f)
        f.close()

    return my_dict


def generate_dict_from_text_file(filename):
    """
    The workhorse of the previous def; takes a single text file and stores its content into a dict.
    The dict is defined using image IDs as keys, and a vector of (label, belief) - tuples as value.
    :param filename: Name of the .txt to read
    :return: The dict
    """
    print('reading ' + filename)
    my_dict = {}
    if os.path.isfile(filename):
        f = open(filename, 'r')

        for line in f.readlines():
            segments = line.rstrip('\n').split(';')

            val = []
            for my_segment in segments[1:]:
                # my_segment contains a word/phrase with a score in parenthesis. Skip element 0, as that is the key.
                parenthesis = my_segment.rfind('(')
                if parenthesis > 0:
                    # We found something
                    val.append(tuple([my_segment[:parenthesis], float(my_segment[parenthesis + 1:-1])]))
            my_dict[segments[0]] = val
        f.close()
        pickle_name = filename.replace('/txt/', '/pickle/').replace('.txt', '.pickle')
        f = open(pickle_name, 'wb')
        pickle.dump(my_dict, f)
        f.close()

    else:
        print('File does not exist:' + filename)

    return my_dict


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-train', default=False, type=bool, help='Train the model before testing', dest='train_model')
    parser.add_argument('-run_1000_samples', default=False, type=bool,
                        help='Only run a 1000 samples instead of full test set',
                        dest='run_1000_samples')
    parser.add_argument('-test_path', default="validate", help='Path to pickle files that should be tested',
                        dest='test_path')
    parser.add_argument('-restore_path', default="./model_checkpoints/", help='Path to pre-trained model',
                        dest='restore_path')

    args = parser.parse_args()

    start_time = time.time()

    # First calls here to make sure we have generated a list of all IDS and their labels stored in a pickle
    # train_labels = generate_dict_from_directory()
    train_labels = pickle.load(open('./train/pickle/combined.pickle', 'rb'))

    # Now, do training -- OPTIONAL
    if args.train_model:
        train()

    # Generate random queries, just to run the "test"-function. These are elements from the TEST-SET folder
    test_labels = pickle.load(open('./' + args.test_path + '/pickle/combined.pickle', 'rb'))
    test_ids = list(test_labels.keys())
    all_labels = {**test_labels, **train_labels}
    no_test_images = len(test_ids)
    queries = []

    if not args.run_1000_samples:
        queries = test_ids  # use this to run the full validation set
    else:
        # Use this to run n random queries
        for i in range(1000):
            queries.append(test_ids[random.randint(0, no_test_images - 1)])

    results = test(queries=queries, location='./' + args.test_path, restore_paht=args.restore_path)

    # Run the score function
    total_score = 0.0
    print(50 * '=' + '\n' + 'Individual image scores:' + '\n' + 50 * '=')
    for image in queries:
        if image in results.keys():
            image_score = score(label_dict=all_labels, target=image, selection=results[image])
        else:
            image_score = 0.0
            print('No result generated for ' + image)
        total_score += image_score
        print('%s scores %8.6f cluster size %d' % (image, image_score, len(results[image])))

    print(50 * '=' + '\n' + 'Average score over %d images: %10.8f' % (len(queries), total_score / len(queries))
          + '\n' + 50 * '=')

    print("Time used:", time.time() - start_time)
