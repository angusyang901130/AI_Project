import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns

import keras
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint

from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split

def load_images():
    train = pd.read_csv('./sign_mnist_train.csv')
    train_labels = train['label'].values
    unique_value = np.array(train_labels)

    train.drop('label', axis = 1, inplace = True)
    train_images = train.values
    train_images = np.array([np.reshape(img, (28, 28)) for img in train_images])
    train_images = np.array([img.flatten() for img in train_images])


    label_binrizer = LabelBinarizer()
    train_labels = label_binrizer.fit_transform(train_labels)
    #train_labels = np_utils.to_categorical(train_labels)

    #plt.imshow(images[0].reshape(28,28))

    x_train, x_test, y_train, y_test = train_test_split(train_images, train_labels, test_size = 0.2, random_state = 84)

    x_train = x_train / 255
    x_test = x_test / 255

    x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
    x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)

    return (x_train, y_train, x_test, y_test)

#plt.imshow(x_train[0].reshape(28,28))
def cnn_mnist():
    x_train, y_train, x_test, y_test = load_images()

    model = Sequential()
    model.add(Conv2D(16, kernel_size=(3,3), activation = 'relu', input_shape=(28, 28 ,1) ))
    model.add(MaxPooling2D(pool_size = (2, 2), strides=(2, 2), padding='same'))
    model.add(Dropout(0.20))

    model.add(Conv2D(32, kernel_size = (3, 3), activation = 'relu'))
    model.add(MaxPooling2D(pool_size = (2, 2), strides=(2, 2), padding='same'))
    model.add(Dropout(0.20))

    model.add(Conv2D(64, kernel_size = (3, 3), activation = 'relu'))
    model.add(MaxPooling2D(pool_size = (2, 2), strides=(2, 2), padding='same'))
    model.add(Dropout(0.20))

    model.add(Flatten())
    model.add(Dense(128, activation = 'relu'))
    model.add(Dropout(0.20))

    model.add(Dense(24, activation = 'softmax'))

    model.compile(loss = keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(), metrics=['accuracy'])
    filepath = "cnn_mnist.h5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    callback_list = [checkpoint]

    model.fit(x_train, y_train, validation_data = (x_test, y_test), epochs=50, batch_size=500, callbacks=callback_list)

    test = pd.read_csv('./sign_mnist_test.csv')
    test_labels = test['label']
    test.drop('label', axis = 1, inplace = True)
    test_images = test.values
    test_images = np.array([np.reshape(img, (28, 28)) for img in test_images])
    test_images = np.array([img.flatten() for img in test_images])
    
    label_binrizer = LabelBinarizer()
    test_labels = label_binrizer.fit_transform(test_labels)
    #test_labels = np_utils.to_categorical(test_labels)
    test_images = test_images.reshape(test_images.shape[0], 28, 28, 1)

    scores = model.evaluate(test_images, test_labels, verbose=0)
    print("CNN Error: %.2f%%" %(100-scores[1]*100))
    model.save('cnn_mnist.h5')

cnn_mnist()