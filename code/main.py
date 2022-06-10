from cnn import train
from load_image import load_images_labels
from set_hand_hist import get_hand_hist
from recognize_gesture import recognize, model_predict
from keras.models import load_model
import numpy as np
import os

if __name__ == "__main__":
    if not os.path.exists('./cnn.h5'):
        load_images_labels()
        train()
        
    model = load_model('cnn.h5')

    print("model has been prepared")
    print("Now set up historgram")

    print("Please press c to appear threshold")
    print("Please make sure all the squares cover your hands")
    print("Please make sure the white part only covers where your skin color is")
    print("If you are satisfied with your historgram, press s to save")

    get_hand_hist()

    print("Do the gesture in the green window")
    model_predict(model, np.zero((50, 50), dtype=np.uint8))
    recognize()
    