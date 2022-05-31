import numpy as np
import pickle
import cv2 as cv
import os
from keras import optimizers
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint
from keras import backend as K

def CNN(image_x, image_y, num_of_class):
    #image_x = 50
    #image_y = 50
    #num_of_class = 44

    model = Sequential()

    #layer 1
    model.add(Conv2D(16, (3, 3), input_shape=(image_x, image_y, 1), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'))
    model.add(Dropout(0.2))

    #layer 2
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'))
    model.add(Dropout(0.2))

    #layer 3
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'))
    model.add(Dropout(0.2))

    #flatten
    model.add(Flatten())

    #output layer
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(num_of_class, activation='softmax'))

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    # not sure which works
    """ model.compile(
        optimizer = optimizers.SGD(lr = 0.01),
        loss = 'categorical_crossentropy',
        metrics = ['accuracy']
    ) """

    model.summary()

    filepath = 'cnn_model.h5'
    checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    callback_list = [checkpoint]
    
    return model, callback_list


