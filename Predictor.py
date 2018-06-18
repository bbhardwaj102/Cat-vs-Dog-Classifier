"""
Created on Sat Mar 31 13:32:03 2018

@author: bbhardwaj
"""

from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K
import cv2
import os
import numpy as np


# dimensions of our images.
img_width, img_height = 150, 150

train_data_dir = 'data/train'
validation_data_dir = 'data/validation'
test_img_path = 'data/test'
nb_train_samples = 2000
nb_validation_samples = 800
epochs = 50
batch_size = 16

if K.image_data_format() == 'channels_first':
    input_shape = (3, img_width, img_height)
else:
    input_shape = (img_width, img_height, 3)

model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=input_shape))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('sigmoid'))

model.load_weights("firstWeights.h5")

# this is the augmentation configuration we will use for testing:
# only rescaling
test_datagen = ImageDataGenerator(rescale=1. / 255)
test_image_generator = test_datagen.flow_from_directory(
    test_img_path,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary')


PATH = os.getcwd()
# Define data path
data_path = PATH + '/data'
data_dir_list = os.listdir(data_path)

img_rows=150
img_cols=150
num_channel=3
num_epoch=20

# Define the number of classes
num_classes = 2

img_data_list=[]
names = ['cat','dog']

while(True):
    img = raw_input('Enter image path : ')
    test_image = cv2.imread(img)
    test_image=cv2.resize(test_image,(150,150))
    test_image = np.array(test_image)
    test_image = test_image.astype('float32')
    test_image /= 255
    print (test_image.shape)
    if num_channel==1:
        if K.image_dim_ordering()=='th':
            test_image= np.expand_dims(test_image, axis=0)
            test_image= np.expand_dims(test_image, axis=0)
            print (test_image.shape)
        else:
            test_image= np.expand_dims(test_image, axis=3)
            test_image= np.expand_dims(test_image, axis=0)
            print (test_image.shape)
    else:
        if K.image_dim_ordering()=='th':
            test_image=np.rollaxis(test_image,2,0)
            test_image= np.expand_dims(test_image, axis=0)
            print (test_image.shape)
        else:
            test_image= np.expand_dims(test_image, axis=0)
            print (test_image.shape)
    
    # Predicting the test image
    pred_class = model.predict_classes(test_image)
    print((model.predict(test_image)))
    print(pred_class)
    print(names[pred_class[0][0]])
    again = raw_input('Do you want to continue (y or n): ')
    if(again == 'n'):
        break
