import cv2
from glob import glob
import numpy as np
import random
from sklearn.utils import shuffle
import pickle
import os

def images_labels_pickle():
    images_labels = []
    images = glob("../gestures/*/*.jpg")
    images.sort()

    for image in images:
        # os.sep => / or \
        # Here we want to label the image with the folder name (number)
        # But, there are 3 os.sep: /gestures \num1 \num2.jpg
        # The number of folder is between second os.sep and the third os.sep 
        label = image[image.find(os.sep)+1: image.rfind(os.sep)]
        img = cv2.imread(image, 0)
        print(image)
        images_labels.append((np.array(img, dtype=np.uint8), int(label)))

    return images_labels


images_labels = images_labels_pickle()
images_labels = shuffle(shuffle(shuffle(shuffle(images_labels))))
#print(images_labels)
images, labels = zip(*images_labels)

# train 
train_images = images[:int(5/6*len(images))]
with open("train_images", "wb") as f:
    pickle.dump(train_images, f)
del train_images

train_labels = labels[:int(5/6*len(images))]
with open("train_labels", "wb") as f:
    pickle.dump(train_labels, f)
del train_labels

# validate
validate_images = images[int(5/6*len(images)):int(11/12*len(images))]
with open("validate_images", "wb") as f:
    pickle.dump(validate_images, f)
del validate_images

validate_labels = labels[int(5/6*len(images)):int(11/12*len(images))]
with open("validate_labels", "wb") as f:
    pickle.dump(validate_labels, f)
del validate_labels

# test
test_images = images[int(11/12*len(images)):]
with open("test_images", "wb") as f:
    pickle.dump(test_images, f)
del test_images

test_labels = labels[int(11/12*len(images)):]
with open("test_labels", "wb") as f:
    pickle.dump(test_labels, f)
del test_labels