import pandas as pd
import numpy as np
import os
import cv2

# Load relevant driving data to dataframe
df = pd.read_csv('data/driving_log.csv',
                 names=['center_path', 'left_path', 'right_path', 'steering', 'throttle', 'brake', 'speed'],
                 usecols=['center_path', 'left_path', 'right_path', 'steering'])

# Center camera training data
X_train_center = np.array([cv2.imread(os.path.join('data', 'IMG', os.path.basename(p))) for p in df.center_path])
y_train_center = df.steering.values

# Center camera horizontal flip
X_train_flip = X_train_center[:, :, ::-1, :]
y_train_flip = -y_train_center

# Left and right cameras
lr_correction = 0.2
X_train_left = np.array([cv2.imread(os.path.join('data', 'IMG', os.path.basename(p))) for p in df.left_path])
X_train_right = np.array([cv2.imread(os.path.join('data', 'IMG', os.path.basename(p))) for p in df.right_path])
y_train_left = y_train_center + lr_correction
y_train_right = y_train_center - lr_correction

# Concatenate datasets
# X_train = np.concatenate((X_train_center, X_train_flip, X_train_left, X_train_right))
# y_train = np.concatenate((y_train_center, y_train_flip, y_train_left, y_train_right))
X_train = np.concatenate((X_train_center, X_train_flip))
y_train = np.concatenate((y_train_center, y_train_flip))

from keras import backend as K
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Dropout
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D


model = Sequential()
model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160, 320, 3)))
model.add(Cropping2D(cropping=((70, 25), (0, 0)))) # crop out top 75 pixels and bottom 25

arch = 'nvidia'

if arch == 'lenet5':

    # LeNet5 Architecture
    model.add(Convolution2D(6, 5, 5, activation='relu'))
    model.add(MaxPooling2D())
    model.add(Convolution2D(6, 5, 5, activation='relu'))
    model.add(MaxPooling2D())
    model.add(Flatten())
    model.add(Dense(120))
    model.add(Dense(84))

elif arch == 'nvidia':

    # Nvidia Architecture
    model.add(Convolution2D(24, 5, 5, subsample=(2, 2), activation='relu'))
    # model.add(Dropout(0.5))
    model.add(Convolution2D(36, 5, 5, subsample=(2, 2), activation='relu'))
    # model.add(Dropout(0.5))
    model.add(Convolution2D(48, 5, 5, subsample=(2, 2), activation='relu'))
    # model.add(Dropout(0.5))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    # model.add(Dropout(0.5))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    # model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(100))
    # model.add(Dropout(0.5))
    model.add(Dense(50))
    # model.add(Dropout(0.5))
    model.add(Dense(10))
    # model.add(Dropout(0.5))

else:

    raise ValueError("Invalid arch. arch must be lenet5 or nvidia")

model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
callbacks = [EarlyStopping(monitor='val_loss', patience=5),
             ModelCheckpoint(filepath='model.h5', verbose=1, save_best_only=True)]
hist = model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=100, callbacks=callbacks)

K.clear_session()