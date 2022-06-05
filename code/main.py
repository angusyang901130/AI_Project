from cnn import train
from load_image import load_images_labels
from keras.models import load_model
import os

if __name__ == "__main__":
    if not os.path.exists('./cnn.h5'):
        load_images_labels()
        train()
    else:
        model = load_model('cnn.h5')

    