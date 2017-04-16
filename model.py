import pandas as pd
import numpy as np
import os
import cv2

# Load relevant driving data to dataframe
df = pd.read_csv('data/driving_log.csv',
                 names=['center_path', 'left_path', 'right_path', 'steering', 'throttle', 'brake', 'speed'],
                 usecols=['center_path','steering'])

# Create training data
X_train = np.array([cv2.imread(os.path.join('data', p.strip())) for p in df.center_path])
y_train = df.steering.values

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D


model = Sequential()
model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160, 320, 3)))
model.add(Convolution2D(6, 5, 5, activation='relu'))
model.add(MaxPooling2D())
model.add(Convolution2D(6, 5, 5, activation='relu'))
model.add(MaxPooling2D())
model.add(Flatten())
model.add(Dense(120))
model.add(Dense(84))
model.add(Dense(1))


model.compile(loss='mse', optimizer='adam')
callbacks = [EarlyStopping(monitor='val_loss', patience=5),
             ModelCheckpoint(filepath='model.h5', verbose=1, save_best_only=True)]
hist = model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=100, callbacks=callbacks)
print(hist.history)

import gc; gc.collect()