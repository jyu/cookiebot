import keras
import tensorflow as tf
from tensorflow.keras.layers import Dense, Lambda, Input, concatenate
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.callbacks import ModelCheckpoint
import numpy as np
from sklearn import preprocessing
import pickle
import argparse
import yaml
import os
from tensorflow.keras.models import load_model

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
loc_dir = config.get('loc_dir')
val_loc_dir = config.get('val_loc_dir')

# Returns data as a python list of np arr features
def getDataFromFile(data_dir, f_name, normalize=True):
    f = open(data_dir + "/" + f_name)
    lines = f.readlines()
    data = []
    for line in lines:
        feat = np.fromstring(line, sep=";")
        if normalize:
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
X_train, X_train_locs, Y_train_x, Y_train_y = [], [], [], []
X_test, X_test_locs, Y_test_x, Y_test_y  = [], [], [], []

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

    y_x = classes_x.index(c_x)
    y_y = classes_y.index(c_y)
    print(y_x, y_y)

    data = getDataFromFile(data_dir, c)
    locs = getDataFromFile(loc_dir, c, normalize=False)
    for i in range(len(data)):
        feat = data[i]
        X_train.append(feat)
        X_train_locs.append(locs[i])
        Y_train_x.append(y_x)
        Y_train_y.append(y_y)

    val_data = getDataFromFile(val_data_dir, c)
    val_locs = getDataFromFile(val_loc_dir, c, normalize=False)
    for i in range(len(val_data)):
        feat = val_data[i]
        X_test.append(feat)
        X_test_locs.append(val_locs[i])
        Y_test_x.append(y_x)
        Y_test_y.append(y_y)

    print("c", c, "data len", len(data), "val data len", len(val_data))

X_train, Y_train_x, Y_train_y = np.array(X_train), np.array(Y_train_x), np.array(Y_train_y)
X_train_locs = np.array(X_train_locs)
X_test, Y_test_x, Y_test_y = np.array(X_test), np.array(Y_test_x), np.array(Y_test_y)
X_test_locs = np.array(X_test_locs)
print(X_train.shape, X_train_locs.shape, Y_train_x.shape, Y_train_y.shape)
print(X_test.shape, X_test_locs.shape, Y_test_x.shape, Y_test_y.shape)
input_shape = X_train[0].shape
input_loc_shape = X_train_locs[0].shape
print(input_loc_shape)
# Model
inp = Input(shape=input_shape, name="input")
inp_loc = Input(shape=input_loc_shape, name="input_loc")
x = concatenate([inp, inp_loc])
x = Dense(16, activation='relu')(x)
x = Dense(16, activation='relu')(x)
x = Dense(16, activation='relu')(x)
x_out = Dense(1, activation='sigmoid')(x)
x_out = Lambda(lambda x: x * 5, name='x_out')(x_out)
y_out = Dense(1, activation='sigmoid')(x)
y_out = Lambda(lambda y: y * 3, name='y_out')(y_out)
model = Model([inp, inp_loc], [x_out, y_out])

model.compile(loss='mean_squared_error', optimizer='adam')
model.summary()        

checkpoint = ModelCheckpoint(model_path, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
model.fit(
    [X_train, X_train_locs],
    [Y_train_x, Y_train_y], 
    batch_size=32, 
    epochs=50, 
    validation_data=([X_test, X_test_locs], [Y_test_x, Y_test_y]),
    shuffle=True,
    callbacks=[checkpoint]
)

# Get train/validation accuracy, based on if correct bin or not
model = load_model(model_path)

names = ['train', 'test']
xs = [X_train, X_test]
xlocs = [X_train_locs, X_test_locs]
yxs = [Y_train_x, Y_test_x]
yys = [Y_train_y, Y_test_y]

for j in range(len(names)):
    x = xs[j]
    loc = xlocs[j]
    yx = yxs[j]
    yy = yys[j]

    res = model.predict([x, loc])
    x_res = res[0]
    y_res = res[1]
    x_correct = 0
    y_correct = 0
    total = x.shape[0]
    for i in range(len(x_res)):
        
        model_x = int(round(x_res[i][0]))
        test_x =  yx[i]
        if model_x == test_x:
            x_correct += 1

        model_y = int(round(y_res[i][0]))
        test_y =  yy[i]
        if model_y == test_y:
            y_correct += 1

    print("X", names[j],  "accuracy", x_correct / total)
    print("Y", names[j], "accuracy", y_correct / total)
