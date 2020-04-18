import keras
import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Model, Sequential
import numpy as np
from sklearn import preprocessing
import pickle
import argparse
import yaml
import os

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("-c", help="config path for nn") # Config name
args = parser.parse_args()
config_path = args.c

# Load parameters from config
config = yaml.load(open(config_path), Loader=yaml.Loader)
model_path = config.get('model_path')
data_dir = config.get('data_dir')
val_data_dir = config.get('val_data_dir')

# Returns data as a python list of np arr features
def getDataFromFile(data_dir, f_name):
    f = open(data_dir + "/" + f_name)
    lines = f.readlines()
    data = []
    for line in lines:
        feat = np.fromstring(line, sep=";")
        feat = preprocessing.scale(feat)
        data.append(feat)
    #print("Found ", len(data), "data for file", f_name)    
    return data

def classToOneHot(i, classes):
    one_hot = np.zeros(len(classes))
    one_hot[i] = 1
    return one_hot

# Get data
classes = os.listdir(data_dir)
X_train, Y_train = [], []
X_test, Y_test = [], []

for i in range(len(classes)):
    c = classes[i]
    y_class = classToOneHot(i, classes)
    data = getDataFromFile(data_dir, c)
    for feat in data:
        X_train.append(feat)
        Y_train.append(y_class)

    val_data = getDataFromFile(val_data_dir, c)
    for feat in val_data:
        X_test.append(feat)
        Y_test.append(y_class)

    print("c", c, "data len", len(data), "val data len", len(val_data))

X_train, Y_train = np.array(X_train), np.array(Y_train)
X_test, Y_test = np.array(X_test), np.array(Y_test)
input_shape = X_train[0].shape

# Model
model = Sequential()
model.add(Dense(8, input_shape=input_shape, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(len(classes), activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()        

model.fit(
    X_train, 
    Y_train, 
    batch_size=32, 
    epochs=100, 
    validation_data=(X_test, Y_test),
    shuffle=True
)

model.save(model_path)
