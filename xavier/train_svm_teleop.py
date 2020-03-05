from sklearn.svm.classes import SVC
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import pickle
import argparse

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('-m') # Model name
args = parser.parse_args()
model_name = args.m

# Returns data as a python list of np arr features
def getDataFromFile(f_name):
    data_dir = "teleop_data"
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
left = getDataFromFile("left")
for l in left:
    X.append(l)
    Y.append(0)
straight = getDataFromFile("straight")
for s in straight:
    X.append(s)
    Y.append(1)
right = getDataFromFile("right")
for r in right:
    X.append(r)
    Y.append(2)

X = np.array(X)
Y = np.array(Y)
print("All data shape X:", X.shape, "Y:", Y.shape)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)


print("All train data shape X_train:", X_train.shape, "Y_train:", Y_train.shape)
print("All test data shape X_test:", X_test.shape, "Y_test:", Y_test.shape)

model = SVC(probability=True, C=10, kernel="rbf")
model.fit(X_train, Y_train)

score = model.score(X_train, Y_train)
print("Training accuracy:", score)
score = model.score(X_test, Y_test)
print("Test accuracy:", score)

out_f = open(model_name, 'wb')
pickle.dump(model, out_f)
