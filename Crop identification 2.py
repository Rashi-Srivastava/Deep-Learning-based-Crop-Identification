#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb  6 02:54:21 2021

@author: rashi
"""

import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, BatchNormalization, Activation
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.constraints import maxnorm
from keras import backend as K 
#K.set_image_dim_ordering('tf')
import pickle
from keras.models import model_from_json
from sklearn.metrics import classification_report, confusion_matrix

from keras.utils import np_utils
import os
from imutils import paths
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator


img_width, img_height = 256, 256

train_data = r"/home/rashi/Downloads/crop/Train"
test_data =  r"/home/rashi/Downloads/crop/Test"

train_path = list(paths.list_images(train_data)) 
totalTrain = len(train_path)

totalTest = len(list(paths.list_images(test_data)))         

trainLabels = [p.split(os.path.sep)[-2] for p in train_path] 
classWeight = dict()


epochs = 25
batch_size = 15

if K.image_data_format() == 'channels_first':
    input_shape = (3, img_width, img_height)
else:
    input_shape = (img_width, img_height, 3)
    
    
model = Sequential(name="ABC")
model.add(Conv2D(32, (3, 3), padding="same",input_shape=(256,256,3)))
model.add(Activation("relu"))

model.add(MaxPooling2D(pool_size=(3, 3)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (4, 4), padding="same"))
model.add(Activation("relu"))

model.add(Conv2D(64, (4, 4), padding="same"))
model.add(Activation("relu"))
#model.add(MaxPooling2D(pool_size=(2, 2)))
#model.add(Dropout(0.25))

model.add(Conv2D(64, (4, 4), padding="same"))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(128, (4, 4), padding="same"))
model.add(Activation("relu"))
#model.add(MaxPooling2D(pool_size=(2, 2)))
#model.add(Dropout(0.25))

model.add(Conv2D(128, (4, 4), padding="same"))
model.add(Activation("relu"))

model.add(Conv2D(128, (4, 4), padding="same"))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(256, (4, 4), padding="same"))
model.add(Activation("relu"))

model.add(Conv2D(256, (4, 4), padding="same"))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

#model.add(Conv2D(1024, (4, 4), padding="same"))
#model.add(Activation("relu"))

model.add(Conv2D(512, (4, 4), padding="same"))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

"""model.add(Conv2D(1024, (4, 4), padding="same"))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))"""

model.add(Flatten())
model.add(Dense(1024))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(10))
model.add(Activation("softmax"))


train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    rotation_range = 40,
    horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(
    train_data,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    shuffle = True,
    color_mode="rgb",
    class_mode='categorical')


test_generator = test_datagen.flow_from_directory(
	test_data,
	class_mode="categorical",
	target_size=(img_width, img_height),
	color_mode="rgb",
	shuffle=False,
	batch_size = batch_size)

optimizer = 'adam'
model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

model.fit_generator(
    train_generator,
    steps_per_epoch=totalTrain // batch_size,
    epochs=epochs,
    class_weight = classWeight)

test_generator.reset()
predIdxs = model.predict(x=test_generator, steps=(totalTest // batch_size) + 1)
#print(predIdxs.shape())
predIdxs = np.argmax(predIdxs, axis=1)

cm = confusion_matrix(test_generator.classes, predIdxs)
total = sum(sum(cm))
acc = (cm[0, 0] + cm[1, 1]) / total
print("acc: {:.4f}".format(acc))

model_json = model.to_json()
with open("model.json", "w") as json_file:
  json_file.write(model_json)
  

import imageio
#import scipy.misc
from matplotlib.pyplot import imshow
#from breast_cancer_classification import train_model 
from keras.preprocessing import image
import numpy as np
#from main.cancernet import CancerNet
from tensorflow.keras.optimizers import Adagrad
from keras.models import model_from_json


image_path= r"/home/rashi/Downloads/crop/Test/Brinjal/Brinjal0.jpg"                          #reading the image received from client
img = image.load_img(image_path)
img = img.resize((150,150))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = x/255.0
    
prediction=(model.predict(x))*100       #predicting soil type based on SoilNET
max_i = np.argmax(prediction)
  

  
'''
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
'''
