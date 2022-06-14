# AI_Project

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
After training, our CNN model is able to predict the 44 character in the ALS with a prediction accuracy > 95%.
