# Dogs and Cats Recognition
  * Input: Dogs or Cats images.
  * Ouput: Classification Dogs or Cats. 
## Abstract
  *This project describes the machine learning method in the binary classification problem that distinguishes the two animals closest to humans: dogs and cats. Using three machine learning methods to perform the problem, including:*
  * *Logistic Regression* 
  * *Neural Network with One Hidden Layer* 
  * *Convolutional Neural Network.*

## Dataset
  Get data form Kaggle
  Image size (64, 64, 3) 
  
*[Follow the link to get data](https://drive.google.com/drive/folders/1m8QMw8JHTn77DefCox0IYVXILwbFN1F2)*

## In-depth 
  ### Logistic Regressios



  ### Neural Network with One Hidden Layer

  
  
  ### Convolutional Neural Network.
> import library
```py
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Dropout
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
```
