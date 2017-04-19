import pandas as pd
import numpy as np
import os
import cv2

fieldnames = ['center_path', 'left_path', 'right_path',
              'steering', 'throttle', 'brake', 'speed']
# Directories of captured data
# left/right camera correction was sufficient so recovery data is excluded
datadirs = ['CenteredCounterClockwise2Laps','CenteredClockwise2Laps']
            #'CounterClockwiseRecovery1Lap', 'ClockwiseRecovery1Lap',
            #'FailureCorrections2']

# Load relevant driving data to dataframes, folder at a time
df_list = []
for datadir in datadirs:
    this_df = pd.read_csv(os.path.join(datadir, 'driving_log.csv'),
                          names=fieldnames,
                          usecols=['center_path', 'left_path',
                                   'right_path', 'steering'])
    this_df['datadir'] = datadir
    for cam in ['center', 'left', 'right']:
        def fixpath(p): return os.path.join(datadir, 'IMG', os.path.basename(p))
        this_df[cam + '_path'] = [fixpath(p.strip()) for p in this_df[cam + '_path']]
    df_list.append(this_df)
df = pd.concat(df_list, ignore_index=True)

# Construct training data, center camera
X_train = np.array([cv2.imread(p) for p in df.center_path])
y_train = df.steering.values

# Left and right camera data with correction factor
lr_correction = 0.4
mask = df.datadir.isin(['CenteredCounterClockwise2Laps', 'CenteredClockwise2Laps'])# == datadirs[0]#')
X_train_left = np.array([cv2.imread(p) for p in df.loc[mask, 'left_path']])
X_train_right = np.array([cv2.imread(p) for p in df.loc[mask, 'right_path']])
y_train_left = df.loc[mask, 'steering'].values + lr_correction
y_train_right = df.loc[mask, 'steering'].values - lr_correction
# Append it
X_train = np.concatenate((X_train_left, X_train_right, X_train))
y_train = np.concatenate((y_train_left, y_train_right, y_train))


from keras import backend as K
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Dropout
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from keras.regularizers import l2

# model parameters
arch = 'nvidia'
keep_prob_conv = 1.0
keep_prob_fc = 0.5
l2_conv = None #l2(0.0)
l2_fc = None #l2(0.0)
loss_func = 'mse'


model = Sequential()
# Normalize the images (max scaling and mean subtraction)
model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160, 320, 3)))
# Crop out top 60 pixels and bottom 25
model.add(Cropping2D(cropping=((60, 25), (0, 0))))

if arch == 'lenet5':

    # LeNet5 Architecture
    model.add(Conv2D(6, 5, 5, activation='relu', W_regularizer=l2_conv))
    model.add(MaxPooling2D())
    model.add(Dropout(keep_prob_conv))
    model.add(Conv2D(6, 5, 5, activation='relu', W_regularizer=l2_conv))
    model.add(MaxPooling2D())
    model.add(Dropout(keep_prob_fc))
    model.add(Flatten())
    model.add(Dense(120, W_regularizer=l2_fc))
    model.add(Dropout(keep_prob_fc))
    model.add(Dense(84, W_regularizer=l2_fc))
    model.add(Dropout(keep_prob_fc))

elif arch == 'nvidia':

    # Nvidia Architecture
    model.add(Conv2D(24, 5, 5, subsample=(2, 2), activation='relu',
                     W_regularizer=l2_conv))
    model.add(Dropout(keep_prob_conv))
    model.add(Conv2D(36, 5, 5, subsample=(2, 2), activation='relu',
                     W_regularizer=l2_conv))
    model.add(Dropout(keep_prob_conv))
    model.add(Conv2D(48, 5, 5, subsample=(2, 2), activation='relu',
                     W_regularizer=l2_conv))
    model.add(Dropout(keep_prob_conv))
    model.add(Conv2D(64, 3, 3, activation='relu',
                     W_regularizer=l2_conv))
    model.add(Dropout(keep_prob_conv))
    model.add(Conv2D(64, 3, 3, activation='relu',
                     W_regularizer=l2_conv))
    model.add(Dropout(keep_prob_fc))
    model.add(Flatten())
    model.add(Dense(100, W_regularizer=l2_fc))
    model.add(Dropout(keep_prob_fc))
    model.add(Dense(50, W_regularizer=l2_fc))
    model.add(Dropout(keep_prob_fc))
    model.add(Dense(10, W_regularizer=l2_fc))
    model.add(Dropout(keep_prob_fc))

else:

    raise ValueError("Invalid arch. arch must be lenet5 or nvidia")

# Final steering regression output layer
model.add(Dense(1))

# Adam optimizer so I don't have to tune learning rate
model.compile(loss=loss_func, optimizer='adam')

# EarlyStopping to prevent overfitting and automatically stop training
# Save best validation loss model
callbacks = [EarlyStopping(monitor='val_loss', patience=5),
             ModelCheckpoint(filepath='model.h5', verbose=1, save_best_only=True)]
# Train the model
hist = model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=100, callbacks=callbacks)

# to deal with weird error message about deleting keras session
K.clear_session()