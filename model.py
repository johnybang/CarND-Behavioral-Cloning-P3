import pandas as pd
import numpy as np
import os
import cv2

# Load driving data to dataframe
df = pd.read_csv('data/driving_log.csv',
                 names=['center_path', 'left_path', 'right_path', 'steering', 'throttle', 'brake', 'speed'])

# Append corresponding image objects to dataframe
for cam in ['center','left','right']:
    df[cam + '_image'] = [cv2.imread(os.path.join('data', p.strip())) for p in df[cam + '_path']]

# Create training data
X_train = np.array(df.center_image.tolist())
y_train = df.speed.values

from keras.models import Sequential
from keras.layers import Flatten, Dense

model = Sequential()
model.add(Flatten(input_shape=(160, 320, 3)))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=7)

model.save('model.h5')

import gc; gc.collect()