# deep_learning_exam

Mikkel Sannes Nylend
mikkelsn@stud.ntnu.no

**To train semantic image query system:**
This trains the semantic image query system (note that this saves the model during training).

`python3 your_code.py`

**To query semantic image query system:**
This sends one image to the system and receives some number of semantically similar images 
(optional to uncomment line 144 in front.py to train the model before restoring).

`python3 front.py`


**To train semantic image clustering system:**
This trains the last deep autoencoder for the clustering system.

`python3 features_autoencoder.py -train`

**To cluster semantic image clustering system:**
This sends a set of images to the system that should be clustered. 

`python3 features_autoencoder.py -cluster_path '/path_to_images/.../'`


**Requirements**
- Python 3.x
- Tensorflow
- Numpy
- PIL
- Matplotlib
- Pickle
