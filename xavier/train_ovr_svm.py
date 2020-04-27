from sklearn.svm.classes import SVC
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import average_precision_score
import pickle
import argparse
import yaml
import os

# OVR SVM is different from train_svm because it trains one SVM for each class
# https://courses.media.mit.edu/2006fall/mas622j/Projects/aisen-project/
# Multiclass ranking SVMs are generally considered unreliable for arbitrary
# data, since in many cases no single mathematical function exists to separate
# all classes of data from one another

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("-c", help="config path for svm") # Config name
args = parser.parse_args()
config_path = args.c

# Load parameters from config
config = yaml.load(open(config_path), Loader=yaml.Loader)
model_path = config.get('model_path')
data_dir = config.get('data_dir')
val_data_dir = config.get('val_data_dir')

# SVM params
C = config.get('C')
kernel = config.get('kernel')
test_size = config.get('test_size')
runs = config.get('runs')
print("C:", C, "| kernel:", kernel, "| test_size:", test_size, "| runs:", runs)

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

classToData = {}
classToValData = {}
classToModel = {}

classes = os.listdir(data_dir)
for c in classes:
    data = getDataFromFile(data_dir, c)
    val_data = getDataFromFile(val_data_dir, c)
    print("c", c, "data len", len(data), "val data len", len(val_data))
    classToData[c] = data
    classToValData[c] = val_data

# Build SVM for each class
for i in range(len(classes)):
    pos_class = classes[i]

    # Get data
    X = []
    Y = []

    for c in classes:
        label = 0

        if c == pos_class:
            label = 1

        data = classToData[c]
        for feat in data:
            X.append(feat)
            Y.append(label)

    
    # Train model
    X = np.array(X)
    Y = np.array(Y)
    #print("All data shape X:", X.shape, "Y:", Y.shape)

    model = SVC(probability=True, C=C, kernel=kernel, class_weight="balanced")
    model.fit(X, Y)
    classToModel[i] = model
    #print(model.classes_)

    score = model.score(X, Y)
    #print("Training accuracy:", score)
    new_model_path = model_path.replace(".svm", "_" + str(i) + ".svm")
    out_f = open(new_model_path, 'wb')
    pickle.dump(model, out_f)

print("Validating model")
# Validation
correct = 0
total = 0
for i in range(len(classes)):
    c = classes[i]

    val_data = classToValData[c]
    for feat in val_data:
        feat = feat.reshape(1, -1)
        scores = []
        for model_name in classToModel:
            model = classToModel[model_name]
            score = model.predict_proba(feat)[0][1]
            scores.append(score)
        scores = np.array(scores)
        best_i = np.argmax(scores)
        if best_i == i:
            correct += 1
        total += 1
    
print("C:", C, "| kernel:", kernel, "| test_size:", test_size, "| runs:", runs)
print("Val accuracy", correct/total)
