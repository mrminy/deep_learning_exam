import pickle
import time
import one_hot_full_trainingset as my_code


def train(location='./train/'):
    """
    The training procedure is triggered here. OPTIONAL to run; everything that is required for testing the model
    must be saved to file (e.g., pickle) so that the test procedure can load, execute and report
    :param location: The location of the training data folder hierarchy
    :return: nothing
    """
    start_time = time.time()
    train_features_path = '1008_features/train_db_features.pickle'

    training_labels = pickle.load(open(location+'pickle/combined.pickle', 'rb'))
    # training_labels = list(training_labels.keys())

    # Training
    net = my_code.DiffNet(training_labels, db_path=location+'pics/*/', db_features_path=train_features_path)
    net.train(training_epochs=10, learning_rate=.003, batch_size=64, save=True, show_cost=True, show_example=True,
              save_path='diffnet3/')
    print("Time used:", time.time() - start_time)


def test(queries=list(), location='./test'):
    """
    Test your system with the input. For each input, generate a list of IDs that is returned
    :param queries: list of image-IDs. Each element is assumed to be an entry in the test set. Hence, the image
    with id <id> is located on my computer at './test/pics/<id>.jpg'. Make sure this is the file you work with...
    :param location: The location of the test data folder hierarchy
    :return: a dictionary with keys equal to the images in the queries - list, and values a list of image-IDs
    retrieved for that input
    """
    # ##### The following is an example implementation -- that would lead to 0 points  in the evaluation :-)
    my_return_dict = {}

    # Load the dictionary with all training files. This is just to get a hold of which
    # IDs are there; will choose randomly among them
    training_labels = pickle.load(open('./train/pickle/combined.pickle', 'rb'))

    # Restoring previously trained net
    train_features_path = '1008_features/train_db_features.pickle'
    net = my_code.DiffNet(training_labels, db_path='./train/pics/*/', db_features_path=train_features_path)
    net.restore('diffnet3/', global_step=10)
    start_time = time.time()
    for query in queries:
        # Finding similar images from db based on query
        cluster = net.query(query, path=location, show_n_similar_imgs=0)
        my_return_dict[query] = cluster

    print("Time used:", time.time() - start_time)

    return my_return_dict
