# add imports here
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import clear_output
import keras
from keras.applications.resnet_v2 import ResNet50V2
from keras.applications.resnet import ResNet152
from keras.layers import Dense, Dropout, Flatten, BatchNormalization, Conv2D, MaxPooling2D
from keras import regularizers
from keras import models
import keras 
from keras.models import load_model
import cv2
from pandas.core.internals.construction import dataclasses_to_dicts

# define additional functions here



def train_predict(X_train, y_train, X_test):
    # check that the input has the correct shape
    # assert X_train.shape == (77220, 1875)
    # assert y_train.shape == (77220, 1)

    # # to test your implementation, the test set should have a shape of (n_test_samples, 1875)
    # assert len(X_test.shape) == 2
    # assert X_test.shape[1] == 1875

    # --------------------------
    # add your data preprocessing, model definition, training and prediction between these lines
    data = X_train
    Target_values = y_train

    print("Rescaling Train Images")
    Images = []
    for d in range(data.shape[0]):
      img = data.iloc[d,:]*255
      img = np.array(img,dtype=np.uint16)
      img = np.resize(img,(25,25,3))
      Images.append(img)
    New_Images = np.asarray(Images)
    print('Sample Input: 1')
    #padded img
    plt.imshow(New_Images[784])
    print('Sample Input: 2')
    #padded img
    plt.imshow(New_Images[6974])
    Target_values = np.asarray(Target_values)
    print("Splitting data into 80:20 percent ratio of train and validation")
    validation_split = int(Target_values.shape[0] * 0.20)
    Xtrain = New_Images[0:validation_split]
    ytrain = Target_values[0:validation_split]
    #--------------
    xval = New_Images[validation_split:]
    yval = Target_values[validation_split:]
    print("X train shape: {}".format(Xtrain.shape))
    print("X validation shape: {}".format(xval.shape))
    print(" target train shape: {}".format(ytrain.shape))
    print(" target validation shape: {}".format(yval.shape))
    print("--"*20, "Building The Deep Learning Model","--"*20)
        
    model = models.Sequential()
    model.add(Conv2D(60, (3, 3), activation='relu', input_shape=(25, 25, 3)))
    model.add(MaxPooling2D())
    model.add(Conv2D(70, (3, 3), activation='relu'))
    model.add(MaxPooling2D())
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(26, activation='softmax'))
    model.compile(optimizer='Adam',
                  loss=keras.losses.SparseCategoricalCrossentropy(),
                  metrics=['accuracy'])

    print("model compiled")
    print(model.summary())
    print("=="*20)
    print("Fitting the MODEL")
    print(ytrain)
    model.fit(Xtrain/255,ytrain, epochs = 2, validation_data=(xval/255, yval), batch_size = 100)
    print("Saving and then loading the model as CharacterRecognizer.h5")
    model.save('CharacterRecognizer.h5')
    model =  load_model('CharacterRecognizer.h5')
    print("Training the TEST data")
    print("Rescaling Train Images")
    Images = []
    dataclasses_to_dicts = X_test/255
    for d in range(data.shape[0]):
      img = data.iloc[d,:]*255
      img = np.array(img,dtype=np.uint16)
      img = np.resize(img,(25,25,3))
      Images.append(img)
    x = np.asarray(Images)
    y_pred = np.argmax(model.predict(x),axis = 1)
    print("Some Prediction: {}".format(y_pred[0:10]))
    



    






    # --------------------------

    # check that the returned prediction has correct shape
    assert y_pred.shape == (len(X_test),)

    return y_pred
print("loading X train data")
data = pd.read_csv('/content/drive/MyDrive/LogoAi/X_train.csv',index_col = 0)
print("Loading Y train data")
Target_values = pd.read_csv('/content/drive/MyDrive/LogoAi/y_train.csv',index_col = 0)
train_predict(data,Target_values,data)

