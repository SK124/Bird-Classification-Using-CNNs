# -*- coding: utf-8 -*-
"""Report UCSD 200.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/gist/SK124/0ea3ce9ff3463f7d04deffe60929e75d/report-ucsd-200.ipynb

### Bird Classification

Dataset Used:Caltech-UCSD Birds 200 (CUB-200) 
It is an image dataset with photos of 200 bird species (mostly North American). 

Number of Categories: 200

Number of Images: 6,033

Annotations: Bounding Box, Rough Segmentation, Attributes
Since we are classsfying we can use only the images and lists files.

Images:648 MB

Lists: with class names, train/test splits, etc.

Usually every class had around 22-36 Images but data augmentation gave more data and decreased the influence of imbalance in data.

So lets move further and go through the different blocks in the notebook.

###Data Extracter Block

Extracting All the required Files.
"""

!tar -xf /content/drive/"My Drive"/"UCSD 200"/images.tgz

!tar -xf /content/drive/"My Drive"/"UCSD 200"/lists.tgz

mkdir bird

mv /content/lists /content/bird

mv /content/images /content/bird

"""###Package Imports Block"""

import numpy as np
import pandas as pd
import os
import keras
from keras.layers import Conv2D, MaxPool2D, Flatten, Dense, Dropout, BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras import regularizers
from sklearn.model_selection import train_test_split
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
from numpy import floor
import random
print(os.listdir("../content"))

data_path='/content/bird/images'

from glob import glob

"""### Dataviewer Block

Visualizing the Data:
"""

def BirdView(genera):
  print("Samples images for Bird " + genera)
  root_path='/content/bird/images/'
  img_path=root_path+genera+'/**'
  images =glob(img_path)
  plt.figure(figsize=(16,16))
  img=random.sample(images,3)
  plt.subplot(131)
  plt.imshow(cv2.imread(img[0]))
  plt.subplot(132)
  plt.imshow(cv2.imread(img[1]))
  plt.subplot(133)
  plt.imshow(cv2.imread(img[2]))
  return

BirdView('043.Yellow_bellied_Flycatcher')

print(os.listdir('/content/bird/images'))

""""._"  files, these invisible files are "resource fork" segments of files that are being created on the external volume.
These were created after running the linux command


These cause the Datagenerator(os.listdir()) function loop to extract all files TWICE since both point to same file, took me a while to figure it out

So I wrote a function to remove such files
"""

import glob


def delete_files(path, pattern):
    for f in glob.iglob(os.path.join(path, pattern)):
        try:
            os.remove(f)
        except OSError as exc:
            print (exc)

"""Note:Its optional to delete them, sometimes when you have limited data it is a uncommmon but useful practice to use dataset again as long as the model does not overfit it is applicable but be careful if you do so."""

delete_files(data_path, '._**')

"""###Label Dictionary Block

So to label all the images during processing and augmentation ,we need a label dictionary to store all the labels, fortunately lists.txt has all classes and coressponding lables, with a simple python program we can exploit this text file to make a python dictionary instead of manually labelling all 200 classes
"""

f = open('/content/bird/lists/classes.txt', 'r')

for line in f:
    v,_ = line.split('.')
    k=line
    labels_dict[k.strip()] = int(v.strip())

f.close()

"""As you can see it stores all the classes and the labels

Take Note that indexing starts from 1 instead of 0 which would be ideal if you use keras's inbuit .to_categorical utility
You can easily do that by making the following change



         labels_dict[k.strip()] = int(v.strip())-1
But it doesnt have any adverse affects, it only increases the columns of one hot encoded labels which I will explain in a moment
"""

labels_dict

"""###Dataloader Block

This Block does the Following:

1.Resizes the image to array of  required size, it uses cv2.INTER_AREA interpolation which is better than other techniques. You can also try cv2.INTER_CUBIC which will keep the features of image constant but crashes colab's memory at size=(256,256) due to complexity of its(cv2.INTER_CUBIC) algorithm


2.Augments the data by flipping 

3.Converts labels to Onehot-Encoded Form
 
Now I create two lists to store images and corresponding labels,
strip function is to remove all the ._ images which we do not want as they are redundant as we process the image we store it in the list and also the labels in its list.
Then we convert image into np.array and labels to to.categorical which makes the labels into one hot encoded form as we are dealing with a multi class classification problem


Then we split the data in following ratio:

X_train=70% 

X_Test=20%

 X_valid=10% 

 You can also follow your own splitting criteria.
"""

def load_data():
  #Loads data and preprocesses it,returns train an test data along with labels
  images=[]
  labels=[]
  
  size=(64,64)
  print('Loaing Data from File ',end='')
  for folder in os.listdir(data_path):
    fol=folder.strip('._')
    path= data_path +  '/' + fol
    print(fol,end='|')
    for image in os.listdir(path):
      try:
         temp_img=cv2.imread(path+'/'+image)
         temp_img=cv2.resize(temp_img,size,interpolation=cv2.INTER_AREA)
         images.append(temp_img)
         labels.append(labels_dict[fol])
         temp_img=cv2.flip(temp_img,flipCode=1)
         images.append(temp_img)
         labels.append(labels_dict[fol])
         
     
      except Exception :
        pass
  images=np.array(images)
  images=images.astype('float32')/255.0
  labels=keras.utils.to_categorical(labels)
  X_train,X_test,y_train,y_test=train_test_split(images,labels,test_size=0.2)
  print()
  print('Loaded',len(X_train),' images for training','Train data shape ',X_train.shape)
  print('Loaded',len(X_test),' images for testing','Test data shape ',X_test.shape)

  return X_train,X_test,y_train,y_test

X_train,X_test,y_train,y_test=load_data()

"""I did try both sizes (64X64) and (128X128) and from my observation they perfrom somewhat similar even though 128X128 size arrays are a bit clearer to interpret to an Human Eye.

Size=64X64
"""

plt.imshow(X_train[12,:,:])
plt.show()

"""Size=128X128"""

plt.imshow(X_train[2101,:,:])
plt.show()

"""###CNN Architecture Block

CUSTOM CNN:

This architecture performed consideraby better than other architectures I experimented.
"""

def create_model():
    model=Sequential()
    model.add(Conv2D(16, kernel_size = [3,3], padding = 'same', activation = 'relu', input_shape = (64,64,3)))
    model.add(Conv2D(32, kernel_size = [3,3], padding = 'same', activation = 'relu'))
    model.add(MaxPool2D(pool_size = [3,3]))
    
    model.add(Conv2D(32, kernel_size = [3,3], padding = 'same', activation = 'relu'))
    model.add(Conv2D(64, kernel_size = [3,3], padding = 'same', activation = 'relu'))
    model.add(MaxPool2D(pool_size = [3,3]))
    
    model.add(Conv2D(128, kernel_size = [3,3], padding = 'same', activation = 'relu'))
    model.add(Conv2D(256, kernel_size = [3,3], padding = 'same', activation = 'relu'))
    model.add(MaxPool2D(pool_size = [3,3]))
    
    model.add(BatchNormalization())
    
    model.add(Flatten())
    model.add(Dropout(0.5))
    model.add(Dense(512, activation = 'relu', kernel_regularizer = regularizers.l2(0.001)))
    model.add(Dense(201, activation = 'softmax'))
    model.compile(optimizer='adam',loss=keras.losses.categorical_crossentropy,metrics=['accuracy'])
    
    print('Model Created')
    model.summary()
    return model

def fit_model():
  model_hist=model.fit(X_train,y_train,batch_size=64,epochs=25,validation_split=0.125)
  return model_hist

"""You can see the last fully connected layer has 201 columns,this is because of the way we labelled the images , keras.to_categorical adds 1 to the number of classes,therefore  200+1= hence 201.

Taken from Keras documentation

Link: https://github.com/keras-team/keras/blob/master/keras/utils/np_utils.py

        num_classes = np.max(y) + 1
        n = y.shape[0]
        categorical = np.zeros((n, num_classes), dtype=dtype)
        categorical[np.arange(n), y] = 1

So in our y_train(one Hot Encoded) we will see that the first column has no values at all beacuse we didnt start labelling with 0 we did it with 1.
As you can see there are no labels in the 1st column.Maximum value in this column is 0 which means no label is being stored.
Hope this ambuigity is clear.

You can also change the way you label but the results will remain identical.
"""

z=y_train[:,0];np.max(z)

z=y_train[:,200];np.max(z)

"""Key Takeaways:

 1.You can see 201 is out of bounds, which means labels starts from 0 and ends at 200 

 2.In our case Label 0 has no image of its kind becuse there are no images of its kind this is the way we constructed our label_dict dictionary.
"""

#z=y_train[:,201];np.max(z)

"""###Training Block"""

model=create_model()
curr_model_hist=fit_model()

model.save_weights('83.55.h5')

"""In the bottom cells I let the model train and checked the result every 25 epochs and saved after every such iteration."""

curr_model_hist=fit_model()

model.save_weights('weights_88.32')
model.save('88.32')

curr_model_hist=fit_model()

curr_model_hist=fit_model()

model.save_weights('weights_89.48.h5')
model.save('model_89.48.h5')

curr_model_hist=fit_model()

model.save("weights_8877.48.h5")

"""### Evaluation Block

Evaluation on Test Set
"""

evaluate_metrics=model.evaluate(X_test,y_test)
print("\nEvaluation Accuracy = ", "{:.2f}%".format(evaluate_metrics[1]*100),"\nEvaluatation Loss = ","{:.6f}%".format(evaluate_metrics[0]))

"""###Miscellaneous Block

Saving the model to disk
"""

model.save(' best_model.h5')
model.save_weights(' best_weights')

model.save_weights(' best_weights.h5')

model.save_weights(' best_weights')

model_yaml = model.to_yaml()
with open(" best_model.yaml", "w") as yaml_file:
    yaml_file.write(model_yaml)
# serialize weights to HDF5
print("Saved model to disk")

"""###Perfomance of 128X128 Images

For this model I changed the image size to 128X128 to check if it gets any better accuracy actually it did not.
Actually it looks like it overfits on training data but we can see valid accuracy is not decreasing but neither increasing.
It seems like it got stuck in a local minima.

Training it any longer maynot improve the perfomance so I stopped training it.
In other notebooks where I experimented **occasionally** it(128X128) produced similar results as size 64X64 one by careful regularization but it's not as stable as 64X64.
"""

model=create_model()
curr_model_hist=fit_model()

"""Saved the test set to disk and put into new folder called 'utils' where I also put save model and weight and uploaded it to the drive for later use.

####Saving Deleiverables to Drive
"""

np.save(' X_test',X_test)

np.save(' y_test',y_test)

mkdir utils

mv /content/' X_test.npy' /content/utils

mv  /content/' y_test.npy' /content/utils

mv  /content/' best_model.h5' /content/utils

mv  /content/' best_weights.h5' /content/utils

mv  /content/' best_weights' /content/utils

mv /content/' best_model.yaml'  /content/utils

from google.colab import drive
drive.mount('/content/gdrive')

"""Zipping and Loading to Drive"""

!pip install -U -q PyDrive

from google.colab import files
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
from google.colab import auth
from oauth2client.client import GoogleCredentials
import zipfile
import os
import sys

zipname = 'delieverables'

def zipfolder(foldername, target_dir):            
    zipobj = zipfile.ZipFile(foldername + '.zip', 'w', zipfile.ZIP_DEFLATED)
    rootlen = len(target_dir) + 1
    for base, dirs, files in os.walk(target_dir):
        for file in files:
            fn = os.path.join(base, file)
            zipobj.write(fn, fn[rootlen:])

zipfolder(zipname, '/content/utils/')

# 1. Authenticate and create the PyDrive client.
auth.authenticate_user()
gauth = GoogleAuth()
gauth.credentials = GoogleCredentials.get_application_default()
drive = GoogleDrive(gauth)

# 2. Create & upload a file text file.
file1 = drive.CreateFile()
file1.SetContentFile(zipname+".zip")
file1.Upload()

"""###Summary and Perfomance of other architectures

As the required accuracy was to be more than 95% I tried out few more exisiting architectures like Resnet18, Resnet50, Vggn16 unfortunaley I could not get any desired accuracy infact lower,maybe because the dataset we were dealing with this too small for big architecures.
 
For these architecture's implementations I used fast.ai library 
I augmented the data to avoid overfitting as much as possible using fast.ai Imagelist

Results:

Vggn16 with batchnorm : Maximum accuracy of 39%
accuracy kept on oscillating back and forth but did not move above 39%

Resnet18: It also did not perform well,got an mere accuracy of 20% and did not converge any further,image size was changed to as described in paper to fit the model

Resnet 50 : It got an accuracy of about 70% but then it did not improve any further.

Suggestions and Questions are Welcome!
"""
