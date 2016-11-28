# deep_learning_exam

Mikkel Sannes Nylend

mikkelsn@stud.ntnu.no

The delivery on Itslearning includes only 10k precomputed labels, but it should be possible to train. 
I would highly recommend downloading the full set of files (including full training set, precomputed labels and pre-trained models) from:

https://drive.google.com/drive/folders/0B9MmSpC3h_u2M1ZfdHloMmdBR3M?usp=sharing


**To train semantic image query system:**
This trains the semantic image query system (note that this saves the model during training).

`python3 your_code.py`

**To query semantic image query system:**
This sends one image to the system and receives some number of semantically similar images 
(optional to uncomment line 144 in front.py to train the model before restoring).

`python3 front.py`

**To train semantic image clustering system:**
This trains the last deep autoencoder for the clustering system.

`python3 features_autoencoder.py -train True`

**To cluster semantic image clustering system:**
This sends a set of images to the system that should be clustered. Switch 'validate' to '<my_folder>' for testing.  

`python3 features_autoencoder.py -test_path 'validate'`

**Requirements**
- Python 3.x
- Tensorflow
- Numpy
- PIL
- Matplotlib
- Pickle
