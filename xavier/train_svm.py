from sklearn.svm.classes import SVC
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import pickle
import argparse
import yaml

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("-c", help="config path for svm") # Config name
args = parser.parse_args()
config_path = args.c

# Load parameters from config
config = yaml.load(open(config_path), Loader=yaml.Loader)
classes = config.get('classes')
model_path = config.get('model_path')
data_dir = config.get('data_dir')
# SVM params
C = config.get('C')
kernel = config.get('kernel')
test_size = config.get('test_size')
print("C:", C, "kernel:", kernel, "test_size:", test_size)

# Returns data as a python list of np arr features
def getDataFromFile(f_name):
    f = open(data_dir + "/" + f_name)
    lines = f.readlines()
    data = []
    for line in lines:
        feat = np.fromstring(line, sep=";")
        feat = preprocessing.scale(feat)
        data.append(feat)
    print("Found ", len(data), "data for file", f_name)    
    return data

# Get data
X = []
Y = []

for i in range(len(classes)):
    c = classes[i]
    data = getDataFromFile(c)
    for feat in data:
        X.append(feat)
        Y.append(i)

X = np.array(X)
Y = np.array(Y)
print("All data shape X:", X.shape, "Y:", Y.shape)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size)

print("All train data shape X_train:", X_train.shape, "Y_train:", Y_train.shape)
print("All test data shape X_test:", X_test.shape, "Y_test:", Y_test.shape)

model = SVC(probability=True, C=C, kernel=kernel)
model.fit(X_train, Y_train)

score = model.score(X_train, Y_train)
print("Training accuracy:", score)
score = model.score(X_test, Y_test)
print("Test accuracy:", score)

out_f = open(model_path, 'wb')
pickle.dump(model, out_f)
