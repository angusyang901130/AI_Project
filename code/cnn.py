import numpy as np
import pickle
import cv2
from glob import glob
from keras import optimizers, metrics
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint
from keras import backend as K

def get_image_size():
    img = cv2.imread('../gestures/1/1.jpg', 0)
    #print(img.shape)
    return img.shape

def get_num_of_class():
    return len(glob('../gestures/*'))

def CNN():
    img_x, img_y = get_image_size()
    num_of_class = get_num_of_class()

    model = Sequential()

    #layer 1
    model.add(Conv2D(16, (3, 3), input_shape=(img_x, img_y, 1), activation='relu'))
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

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy', metrics.categorical_accuracy])
    # not sure which works
    """ model.compile(
        optimizer = optimizers.SGD(lr = 0.01),
        loss = 'categorical_crossentropy',
        metrics = ['accuracy']
    ) """

    filepath = 'cnn.h5'
    checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    callback_list = [checkpoint]
    
    return model, callback_list

def train():
    
    with open('train_images', 'rb') as f:
        train_images = np.array(pickle.load(f))
    with open('train_labels', 'rb') as f:
        train_labels = np.array(pickle.load(f), dtype=np.int32)
    with open('validate_images', 'rb') as f:
        validate_images = np.array(pickle.load(f))
    with open('validate_labels', 'rb') as f:
        validate_labels = np.array(pickle.load(f), dtype=np.int32)
    with open('test_images', "rb") as f:
        test_images = np.array(pickle.load(f))
    with open('test_labels', 'rb') as f:
        test_labels = np.array(pickle.load(f), dtype=np.int32)

    img_x, img_y = get_image_size()

    train_images = np.reshape(train_images, (train_images.shape[0], img_x, img_y, 1))
    validate_images = np.reshape(validate_images, (validate_images.shape[0], img_x, img_y, 1))
    #test_images = np.reshape(test_images, (test_images.shape[0], img_x, img_y, 1))

    # transfer integer to binary encoding, example: num_of_class = 3, 2 => [0,0,1], 1 => [0,1,0]
    # one hot encoding
    train_labels = np_utils.to_categorical(train_labels)
    validate_labels = np_utils.to_categorical(validate_labels)
    #test_labels = np_utils.to_categorical(test_labels)
    
    model, callback_list = CNN()
    model.summary()
    model.fit(train_images, train_labels, validation_data=(validate_images, validate_labels), epochs=20, batch_size=500, callbacks=callback_list)
    #print("evaluate on test dataset")
    scores = model.evaluate(validate_images, validate_labels, verbose=0)
    print("CNN Error: %.2f%%" % (100-scores[1]*100))
    model.save('cnn.h5')
