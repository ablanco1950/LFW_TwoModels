# -*- coding: utf-8 -*-
"""


by Alfonso Blanco García , April 2022
"""

######################################################################
# PARAMETERS
######################################################################
dirname = "C:\\lfw3"
dirname_test = "C:\\lfw3_test"

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
                gray1=cv2.resize(gray, (250,250))
                
                gray1=gray1[35:175, 35:175]
                
                for y in range (140):
                    lx=gray1[y]
                    for z in range (140):
                        if lx[z] < 100:
                           lx[z]=0
                        #negativ image
                        #lx[z] = 255 -  lx[z]  
                    gray1[y]=lx
                
                gray1 =gray1.flatten() 
                                            
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



X_test, Y_test, TabNumImage_test, TabDenoClass_test = loadimages(dirname_test)


print( "")
for i in range(len(Y_train)):
    print(TabNumImage[i]+ " is class " + str(Y_train[i]))
print("")

num_classes=len(TabDenoClass)
print("Number of classes = " + str(num_classes))
x_train=np.array(X_train)

y_train=np.array(Y_train)

x_test=np.array(X_test)
y_test=np.array(Y_test)

from sklearn.svm import SVC
import pickle #to save the model

from sklearn.multiclass import OneVsRestClassifier


#https://scikit-learn.org/stable/modules/generated/sklearn.multiclass.OneVsRestClassifier.html
model = OneVsRestClassifier(SVC(kernel='linear', probability=True, verbose=True, max_iter=1000)) #Creates model instance here
model.fit(x_train, y_train) #fits model with training data

pickle.dump(model, open("./model.pickle", 'wb')) #save model as a pickled file

predictions = model.predict(x_test)

TotalHits=0
TotalFailures=0

   
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

