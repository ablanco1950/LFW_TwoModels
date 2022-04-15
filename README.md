# LFW_TwoModels
A recognition process of images contained in the LFW database http://vis-www.cs.umass.edu/lfw/#download is carried out using two models, one based on the minimum distance between training image records and test and another that is an adaptation of the CNN KERAS model https://keras.io/examples/vision/mnist_convnet/. Both models are complementary.

Requirements:

Have python installed (not necessarily the latest version, although it is the one with which the tests have been carried out) and the packages corresponding to:
import os

import re

import cv2

import numpy as np

from tensorflow import keras

from tensorflow.keras import layers

from scipy.spatial.distance import cdist

import time

It is recommended to have Anconda installed and to work with Spyder from Anaconda, which guarantees an integrated and friendly environment, installing any missing package from the Anaconda cmd.exe prompt option with commands such as:

python -m pip install keras (case of  keras)

python -m pip install opencv-python (case of  cv2)

python -m pip install tensorflow (case of  tensorflow)


Functioning:

are accompanied the test and training datasets

lfw3.zip containing training images downloaded from http://vis-www.cs.umass.edu/lfw/#download  and specifically from the option All images aligned with commercial face alignment software (LFW-a - Taigman, Wolf, Hassner) 

lfw3_test.zip containing the test images used.No image of the test is found in the training, this is important because the model based on CNN gives extremely good results when the image to be tested is in the training and it drops a lot using images in the test that are not in the training. In the model based on minimum distance, it is avoided by means of an instruction that does not consider the images whose minimum distance is zero.

Both files should be downloaded to the C: drive, otherwise you have to change the dirname and dirname_test parameters at the beginning of the two attached programs.

Note that both files contain images from the lfw2 image database, referenced above, but removing interfering images.That is, a necessary debugging of images has been carried out.

The programs to run are:

GuessImagesLFW_MinDist.py which performs the test according to the minimum distance model.
GuessImagesLFW_CnnKeras.py which performs the test according to the cnn keras model.

It is important to point out that both models seem complementary, when executing GuessImagesLFW_MinDist.py the 7 errors detailed in the attached table appear, which appear corrected when executing the program GuessImagesLFW_CnnKeras and vice versa.


Images failured with minimun distance cCorrected by CNN KERAS, Yes/No
 
Angelina_Jolie_0010.jpg:  Yes
 
Angelina_Jolie_0019.jpg: Yes

Arnold_Schwarzenegger_0007.jpg: Yes 

Arnold_Schwarzenegger_0008.jpg: Yes

Bill_Clinton_0002.jpg: Yes

Bill_Clinton_0003.jpg: Yes

Bill_Clinton_0004.jpg: es

The program GuessImagesLFW_MinDist.py allows to detect the image that has been predicted as having the minimum distance, which is important to detect the images that produce errors or interferences and allows the necessary debugging of the image database.

References:

https://keras.io/examples/vision/mnist_convnet/.

http://vis-www.cs.umass.edu/lfw/#download

https://stackoverflow.com/questions/1871536/minimum-euclidean-distance-between-points-in-two-different-numpy-arrays-not-wit

https://github.com/ablanco1950/MNIST_WITHOUT_SKLEARN

https://www.aprendemachinelearning.com/clasificacion-de-imagenes-en-python/

https://victorzhou.com/blog/intro-to-cnns-part-1/

https://victorzhou.com/blog/intro-to-cnns-part-2/

https://victorzhou.com/blog/keras-cnn-tutorial/

https://realpython.com/face-recognition-with-python/

https://realpython.com/face-detection-in-python-using-a-webcam/


