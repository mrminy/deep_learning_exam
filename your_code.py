import argparse
import pickle
import time
import one_hot_full_trainingset as my_code


def train(location='./train/', save_model=True, small_training_set=False, show_example=False):
    """
    The training procedure is triggered here. OPTIONAL to run; everything that is required for testing the model
    must be saved to file (e.g., pickle) so that the test procedure can load, execute and report
    :param location: The location of the training data folder hierarchy
    :param save_model: save the model or not (saves to 'model_checkpoints/')
    :param small_training_set: train with a small training set (10k images). Because of limitations in Itslearning)
    :param show_example: show a query example at the end of training
    :return: nothing
    """
    start_time = time.time()
    train_features_path = '1008_features/train_db_features.pickle'

    training_labels = pickle.load(open(location + 'pickle/combined.pickle', 'rb'))
    # training_labels = list(training_labels.keys())

    # Training
    net = my_code.DiffNet(training_labels, db_path=location + 'pics/*/', db_features_path=train_features_path)

    # When small_training_set=True it means that only 10k images are used during training (because of limitations in Itslearning)
    net.train(training_epochs=10, learning_rate=.003, batch_size=64, save=save_model, show_cost=False,
              show_example=show_example, save_path='model_checkpoints/', small_training_set=small_training_set)
    print("Time used:", time.time() - start_time)


def test(queries=list(), location='./test', restore_paht='./model_checkpoints/'):
    """
    Test your system with the input. For each input, generate a list of IDs that is returned
    :param queries: list of image-IDs. Each element is assumed to be an entry in the test set. Hence, the image
    with id <id> is located on my computer at './test/pics/<id>.jpg'. Make sure this is the file you work with...
    :param location: The location of the test data folder hierarchy
    :return: a dictionary with keys equal to the images in the queries - list, and values a list of image-IDs
    retrieved for that input
    """
    my_return_dict = {}

    # Load the dictionary with all training files. This is just to get a hold of which
    # IDs are there; will choose randomly among them
    training_labels = pickle.load(open('./train/pickle/combined.pickle', 'rb'))

    # Restoring previously trained net
    net = my_code.DiffNet(training_labels, db_path='./train/pics/*/')
    net.restore(restore_paht)
    start_time = time.time()
    for i, query in enumerate(queries):
        # Finding similar images from db based on query
        if i + 1 % 100 == 0:
            print("Checking", i, "of", len(queries), "- img:", query)
        cluster = net.query(query, path=location + '/pics/*/', show_n_similar_imgs=0)
        my_return_dict[query] = cluster

    print("Time used:", time.time() - start_time)

    return my_return_dict


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-small_training_set', default=False, type=bool,
                        help='If you do not have all 100k labels, this must be True!', dest='small_training_set')
    parser.add_argument('-save_model', default=True, type=bool, help='Save the model or not', dest='save_model')
    parser.add_argument('-show_examples', default=False, type=bool, help='Show examples at the end of training or not',
                        dest='show_examples')

    args = parser.parse_args()

    train(save_model=args.save_model, small_training_set=args.small_training_set, show_example=args.show_examples)
