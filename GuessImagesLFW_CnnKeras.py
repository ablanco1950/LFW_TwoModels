# -*- coding: utf-8 -*-
"""
Adapted from https://keras.io/examples/vision/mnist_convnet/ to images

database lfw http://vis-www.cs.umass.edu/lfw/#download

by Alfonso Blanco García , March 2022
"""

######################################################################
# PARAMETERS
######################################################################
dirname = "C:\\lfw3"
dirname_test = "C:\\lfw3_test"
#dirname = "C:\\lfw2"
#dirname_test=dirname
batch_size = 128
epochs = 30
######################################################################

import os
import re

import cv2

import numpy as np

#########################################################################
def loadimages (dirname ):
#########################################################################
# adapted from:
#  https://www.aprendemachinelearning.com/clasificacion-de-imagenes-en-python/
# by Alfonso Blanco García
########################################################################  
    imgpath = dirname + "\\"
    
    images = []
    directories = []
    dircount = []
    prevRoot=''
    cant=0
    
    print("Reading imagenes from ",imgpath)
    NumImage=-2
    
    Y=[]
    TabNumImage=[]
    TabDenoClass=[]
    for root, dirnames, filenames in os.walk(imgpath):
        
        NumImage=NumImage+1
        
        for filename in filenames:
            
            if re.search("\.(jpg|jpeg|png|bmp|tiff)$", filename):
                cant=cant+1
                filepath = os.path.join(root, filename)
                # https://stackoverflow.com/questions/51810407/convert-image-into-1d-array-in-python
                
                image = cv2.imread(filepath)
               
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
               
                
                gray1=gray[35:175, 35:175]
                
                for y in range (140):
                    lx=gray1[y]
                    for z in range (140):
                        if lx[z] < 100:
                           lx[z]=0
                        
                    gray1[y]=lx
                
                                              
                images.append(gray1)
                if NumImage < 0:
                    NumImage=0
                Y.append(NumImage)
               
               
                TabNumImage.append(filename)
                # b = "Leyendo..." + str(cant)
                #print (b, end="\r")
                if prevRoot !=root:
                  
                    prevRoot=root
                    directories.append(root)
                    dircount.append(cant)
                    cant=0
                    #print ("FILENAME " + filenames[0])
                    #TabDenoClass.append(filenames[0])
                    DenoClass=filenames[0]
                    DenoClass=DenoClass[0:len(DenoClass)-9]
                    
                    
                    TabDenoClass.append(DenoClass)
    print("")
    print('directories read:',len(directories))
    
    print('Total sum of images in subdirs:',sum(dircount))
    
    return images, Y, TabNumImage, TabDenoClass
 
###########################################################
# MAIN
##########################################################


X_train, Y_train, TabNumImage, TabDenoClass = loadimages (dirname )

#for i in range(len(Y)):
#    print(TabNumImage[i]+ " is class " + str(Y[i]))
#print("")

X_test, Y_test, TabNumImage_test, TabDenoClass_test = loadimages(dirname_test)

x_train=np.array(X_train)
y_train=np.array(Y_train)

x_test=np.array(X_test)
y_test=np.array(Y_test)
pp_test=y_test


import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
# Model / data parameters

num_classes=len(TabDenoClass)
print("Numero de Clases = " + str(num_classes))
input_shape = (140, 140, 1)

# the data, split between train and test sets
#(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# Scale images to the [0, 1] range
x_train = x_train.astype("float32") / 255
x_test = x_test.astype("float32") / 255
# Make sure images have shape (140, 140, 1)
x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)
print("x_train shape:", x_train.shape)
print(x_train.shape[0], "train samples")
print(x_test.shape[0], "test samples")


# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

model = keras.Sequential(
    [
        keras.Input(shape=input_shape),
        layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
        #layers.MaxPooling2D(pool_size=(2, 2)),
        #https://medium.com/imagescv/all-about-pooling-layers-for-convolutional-neural-networks-cnn-c4bca1c35e31
        layers.AveragePooling2D(pool_size=(2, 2)),
        layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
        #layers.MaxPooling2D(pool_size=(2, 2)),
        #https://medium.com/imagescv/all-about-pooling-layers-for-convolutional-neural-networks-cnn-c4bca1c35e31
        layers.AveragePooling2D(pool_size=(2, 2)),
        #layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
        
        #layers.MaxPooling2D(pool_size=(2, 2)),
        #https://medium.com/imagescv/all-about-pooling-layers-for-convolutional-neural-networks-cnn-c4bca1c35e31
        #layers.AveragePooling2D(pool_size=(2, 2)),
        layers.Flatten(),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation="softmax"),
    ]
)

model.summary()

model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.1)

test_scores = model.evaluate(x_test, y_test, verbose=2)
print("Test loss:", test_scores[0])
print("Test accuracy:", test_scores[1])
score = model.evaluate(x_test, y_test, verbose=0)
print("Test loss:", score[0])
print("Test accuracy:", score[1])


predictions = model.predict(x_test)

predictions=np.argmax(predictions, axis=1)


TotalHits=0
TotalFailures=0

#for j in range(len(TabDenoClass)):
#   print (str(j) + " " + TabDenoClass[j])
   
print("")
print("List of successes/errors:")       
for i in range(len(x_test)):
    DenoClass=TabNumImage_test[i]
    DenoClass=DenoClass[0:len(DenoClass)-9]
    if DenoClass!=TabDenoClass[(predictions[i])]:
        TotalFailures=TotalFailures + 1
        print("ERROR " + TabNumImage_test[i]+ " is assigned class " + str(predictions[i])
              + " " + TabDenoClass[(predictions[i])] )
              
    else:
        print(TabNumImage_test[i]+ " is assigned class " + str(predictions[i])
              + " " + TabDenoClass[(predictions[i])])
        TotalHits=TotalHits+1
           
print("")
print("Total hits = " + str(TotalHits))  
print("Total failures = " + str(TotalFailures) )     
print("Accuracy = " + str(TotalHits*100/(TotalHits + TotalFailures)) + "%") 

