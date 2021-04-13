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

TRAIN_DIR = './train/'
TEST_DIR = './test/'

ROWS = 64
COLS = 64
CHANNELS = 3

train_images = [TRAIN_DIR + i for i in os.listdir(TRAIN_DIR)]
test_images = [TEST_DIR + i for i in os.listdir(TEST_DIR)]


def read_image(file_path):
    img = cv2.imread(file_path, cv2.IMREAD_COLOR)
    return cv2.resize(img, (ROWS, COLS), interpolation=cv2.INTER_CUBIC)


def prep_data(images):
    m = len(images)
    X = np.ndarray((m, ROWS, COLS, CHANNELS), dtype=np.uint8)
    y = np.zeros((m, 1))
    print("X.shape is {}".format(X.shape))

    for i, image_file in enumerate(images):
        image = read_image(image_file)
        X[i, :] = np.squeeze(image.reshape((ROWS, COLS, CHANNELS)))
        if 'dog' in image_file.lower():
            y[i, 0] = 1
        elif 'cat' in image_file.lower():
            y[i, 0] = 0
        else:  # for test data
            y[i, 0] = image_file.split('/')[-1].split('.')[0]

        if i % 5000 == 0:
            print("Proceed {} of {}".format(i, m))

    return X, y


X_train, y_train = prep_data(train_images)
X_test, y_test = prep_data(test_images)
print("Train shape: {}".format(X_train.shape))
print("Test shape: {}".format(X_test.shape))
X, y = shuffle(X_train, y_train)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=1)
print("Train shape: {}".format(X_train.shape))
print("Train label shape: {}".format(y_train.shape))
print("Validation shape: {}".format(X_val.shape))
print("Validation label shape: {}".format(y_val.shape))
y_train_one_hot = to_categorical(y_train)
print(y_train_one_hot.shape)
num_classes = y_train_one_hot.shape[1]
print(num_classes)
y_val_one_hot = to_categorical(y_val)
print(y_val_one_hot.shape)
classes = {0: 'cats',
           1: 'dogs'}
X_train_norm = X_train / 255
X_val_norm = X_val / 255

model1 = Sequential()

model1.add(Conv2D(32, (3,3), input_shape=(ROWS, COLS, CHANNELS), activation='relu'))
model1.add(MaxPooling2D(pool_size = (2,2)))

model1.add(Conv2D(32, (3,3), activation='relu'))
model1.add(MaxPooling2D(pool_size = (2,2)))
model1.add(Dropout(0.6))

model1.add(Conv2D(64, (3,3), activation='relu'))
model1.add(MaxPooling2D(pool_size = (2,2)))

model1.add(Conv2D(64, (3,3), activation='relu'))
model1.add(MaxPooling2D(pool_size = (2,2)))

model1.add(Flatten())
model1.add(Dropout(0.6))

model1.add(Dense(units=120, activation='relu'))
model1.add(Dense(units=2, activation='sigmoid'))
model1.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model1.summary()
history = model1.fit(X_train_norm, y_train_one_hot, validation_data=(X_val_norm, y_val_one_hot), epochs=20, batch_size=64)
print(history.history.keys())
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
