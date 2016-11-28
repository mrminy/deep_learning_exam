from sklearn.manifold import TSNE
from inception import ImageFeatureExtractor
from front import generate_dict_from_directory

test_labels = generate_dict_from_directory(pickle_file='./validate/pickle/combined.pickle', directory='./validate/txt/')
test_ids = list(test_labels.keys())
feature_extractor = ImageFeatureExtractor()  # Inception-v3
X = feature_extractor.run_inference_on_images(test_labels, path='validate/pics/*/')

tsne = TSNE(learning_rate=100)
X_tsne = tsne.fit_transform(X)


print(X_tsne)
print(len(X_tsne))
