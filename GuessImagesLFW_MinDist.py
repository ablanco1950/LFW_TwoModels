# -*- coding: utf-8 -*-
"""
Guess images from database lfw http://vis-www.cs.umass.edu/lfw/#download
depured using minimum distance between test and training

Author:  Alfonso Blanco García , March 2022
"""

######################################################################
# PARAMETERS
######################################################################
dirname = "C:\\lfw3"
dirname_test = "C:\\lfw3_test"


# https://www.aprendemachinelearning.com/clasificacion-de-imagenes-en-python/
import os
import re

import cv2


from scipy.spatial.distance import cdist
import time
import numpy as np

#########################################################################
# adapted from:
#  https://www.aprendemachinelearning.com/clasificacion-de-imagenes-en-python/
# by Alfonso Blanco García
#########################################################################
def loadimages (dirname ):
    
    imgpath = dirname + "\\"
    
    images = []
    directories = []
    dircount = []
    prevRoot=''
    cant=0
    
    print("Reading images from de ",imgpath)
    NumImage=0.0
    Y=[]
    TabNumImage=[]
    TabDenoClass=[]
    for root, dirnames, filenames in os.walk(imgpath):
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
                
                lgray=gray1.flatten()
                images.append(lgray)
                Y.append(NumImage)
                NumImage=NumImage+1
               
                TabNumImage.append(filepath)
                DenoClass=filenames[0]
                DenoClass=DenoClass[0:len(DenoClass)-9]
                
                
                TabDenoClass.append(DenoClass)
                
                
                if prevRoot !=root:
                  
                    #print(root, cant)
                    prevRoot=root
                    directories.append(root)
                    dircount.append(cant)
                    cant=0
    
    print('Directories read::',len(directories))
    
    #print('Total of images in subdirs:',sum(dircount))
    return images, Y, TabNumImage, TabDenoClass
 
###########################################################
# MAIN
##########################################################

X_train, Y_train, TabNumImage, TabDenoClass = loadimages (dirname )

X_test, Y_test, TabNumImage_test, TabDenoClass_test = loadimages(dirname_test)



##########################################################################333
#   https://stackoverflow.com/questions/1871536/minimum-euclidean-distance-between-points-in-two-different-numpy-arrays-not-wit
############################################################################
TotAciertos=0
TotFallos=0
Inicio=time.time()
DifeMatriz =cdist( X_test, X_train )


for i in range (len(X_test) ):  
    DistanciaMax=999999999999
    LineaMax=-1
    DifeFila=DifeMatriz[i]
    
    j=-1
    while j < len(X_train)-1:
         
          j=j+1
         
          
          DistanciaFila=DifeFila[j]
          
          # if you don't put and (DistanciaFila != 0), in the case
          #of an image that was in both the training and the test
          #it would give zero distance. Should not be considered as a hit
          
          if (DistanciaFila < DistanciaMax) and (DistanciaFila != 0):
        
              DistanciaMax = DistanciaFila
              LineaMax=j
    #print(TabNumImage[j] + "distancia minima =" + str(Distancia_minima))
    print("")
    print("----------------------------------------------------------")
    print("Predicted image : " + TabNumImage[LineaMax] + " - distancia minima =" + str(DistanciaMax))
    print("")    
    print("image to test : " + TabNumImage_test[i] ) 
    
    
    DenoClass=TabNumImage_test[i]
    
    pos=-1
    for k in range (len(DenoClass)):
        if DenoClass[k]=="\\":
            pos=k
        
    DenoClass = DenoClass [pos+1:len(DenoClass)-9] 
    
    print(DenoClass)           
    if TabDenoClass[LineaMax]==DenoClass:
         TotAciertos=TotAciertos+1
    else:
         TotFallos=TotFallos+1
print("") 
print("Total Hits = " + str(TotAciertos ))
print("Total Failures = " + str(TotFallos))  
print("") 
Fin =time.time()  
print ( "procesing time  " +  str(Fin - Inicio))
