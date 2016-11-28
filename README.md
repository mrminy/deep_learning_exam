# deep_learning_exam

Mikkel Sannes Nylend

mikkelsn@stud.ntnu.no

The delivery on Itslearning includes only the code. 
To reproduce and test, I would highly recommend downloading the full set of files 
(including full training set, precomputed labels and pre-trained models) from:

https://drive.google.com/drive/folders/0B9MmSpC3h_u2M1ZfdHloMmdBR3M?usp=sharing


**To train semantic image query system:**
This trains the semantic image query system (note that this saves the model during training).

`python3 your_code.py`


**To query semantic image query system:**
This sends one image to the system and receives some number of semantically similar images. 
If no pre-trained model exists, this will not be possible without training.
Switch 'validate' to '<my_folder>' for testing.

`python3 front.py -test_path 'validate' -train False`

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
