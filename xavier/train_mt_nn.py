import keras
import tensorflow as tf
from tensorflow.keras.layers import Dense, Lambda, Input
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.callbacks import ModelCheckpoint
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

def classToTuple(c):
    c_split = c.split("_")
    c_x = c_split[1]
    c_y = c_split[2]

    if "n" in c_x:
        c_x = int(c_x[1]) * -1
    else:
        c_x = int(c_x)
    c_y = int(c_y)
    return (c_x, c_y)

# Get data
classes = os.listdir(data_dir)
X_train, Y_train_x, Y_train_y = [], [], []
X_test, Y_test_x, Y_test_y  = [], [], []

classes_x = []
classes_y = []
for i in range(len(classes)):
    c = classes[i]
    c_x, c_y = classToTuple(c)
    if not c_y in classes_y:
        classes_y.append(c_y)
    if not c_x in classes_x:
        classes_x.append(c_x)

classes_x = sorted(classes_x)
classes_y = sorted(classes_y)
print(classes_x)
print(classes_y)

for i in range(len(classes)):
    c = classes[i]
    c_split = c.split("_")
    
    c_x, c_y = classToTuple(c)

    y_x = classToOneHot(classes_x.index(c_x), classes_x)
    y_y = classToOneHot(classes_y.index(c_y), classes_y)

    data = getDataFromFile(data_dir, c)
    for feat in data:
        X_train.append(feat)
        Y_train_x.append(y_x)
        Y_train_y.append(y_y)

    val_data = getDataFromFile(val_data_dir, c)
    for feat in val_data:
        X_test.append(feat)
        Y_test_x.append(y_x)
        Y_test_y.append(y_y)

    print("c", c, "data len", len(data), "val data len", len(val_data))

X_train, Y_train_x, Y_train_y = np.array(X_train), np.array(Y_train_x), np.array(Y_train_y)
X_test, Y_test_x, Y_test_y = np.array(X_test), np.array(Y_test_x), np.array(Y_test_y)
input_shape = X_train[0].shape

# Model
inp = Input(shape=input_shape, name="input")
x = Dense(8, activation='relu')(inp)
x = Dense(8, activation='relu')(x)
x_out = Dense(3, activation='softmax', name='x_out')(x)
y_out = Dense(3, activation='softmax', name='y_out')(x)
model = Model(inp, [x_out, y_out])

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()        

path = model_path.replace(".h5", "_mt.h5")
checkpoint = ModelCheckpoint(path, monitor='val_y_out_accuracy', verbose=1, save_best_only=True, mode='max')

model.fit(
    X_train, 
    [Y_train_x, Y_train_y], 
    batch_size=32, 
    epochs=100, 
    validation_data=(X_test, [Y_test_x, Y_test_y]),
    shuffle=True,
    callbacks=[checkpoint]
)

