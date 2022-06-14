# AI_Project

## Enviornment
Python 3.10.4

## Requirements
0. pandas
1. numpy
2. scikit-learn
3. sklearn
4. keras
5. opencv-python
6. tensorflow
7. pyttsx3
8. matplotlib
9. seaborn

## Goal
We want to compare CNN model prediction between different kind of dataset and see if we can get a better 

prediction, and then use as the model for recognizing sign language. 

We also want to reduce the complexity of the setting dataset to realize the instant interpreting. 

After testing, we want to apply the method to instant recognition.

## Guidence

### Step 1: Start the program
Go to folder code and open it in terminal

`python main.py`  
  
### Step 2: Set up hand histogram
When the window of camera image shows, put your palm to cover all green squares and press c to check if your hand could be detected and become white color.

if you are not satisified with the result of shape of hand, press c to readjust the recognized shape

Press s to save the hist

### Step 3: Recognize
When the window with green square appears, do hand gesture in the green square, it would then predict what alphabet or word it might be and add to the text.

If you want to add space, press space.

### Step 4-1: Modification
If you want to modify the text, press m and then press d to delete the letter one at a time. 

If you finish modifying, press q to quit modification and start doing the next gesture.

### Step 4-2: Say out the text
If you finish doing the gestures, press s to say out the text.

### Step 5: Finish program
Press q to finish program

## Result
### Comparison between 2 kinds of dataset
Comparing the result of training 20 epochs with input of no blurred dataset and input of blurred dataset,
the former CNN model has a prediction accuracy about 90%, and the latter CNN model has a prediction accuracy > 95%.

This comparison has quite equal size of dataset and the dataset includes data of A to Z without J and Z.

### Interpreting
We use the blurred images as input to train our model, this time the dataset consists of J and Z and other words such as love, 

our CNN model is able to predict the 44 characters or words in the ALS with a prediction accuracy > 95%.

The setup of environment is quite important, if the noise is big, the accuracy of setting hand histogram would decrease.

This is our prediction and recognition example

![image](https://user-images.githubusercontent.com/90640506/173620865-1212596d-c667-4a16-8598-2d53b2aa9e42.png)

