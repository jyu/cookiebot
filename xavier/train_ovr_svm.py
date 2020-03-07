from sklearn.svm.classes import SVC
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import average_precision_score
import pickle
import argparse
import yaml

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
classes = config.get('classes')
model_path = config.get('model_path')
data_dir = config.get('data_dir')
# SVM params
C = config.get('C')
kernel = config.get('kernel')
test_size = config.get('test_size')
runs = config.get('runs')
print("C:", C, "| kernel:", kernel, "| test_size:", test_size, "| runs:", runs)

# Returns data as a python list of np arr features
def getDataFromFile(f_name):
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
for c in classes:
    data = getDataFromFile(c)
    classToData[c] = data

# Build SVM for each class
accs = np.zeros((runs, len(classes)))
mAPs = np.zeros((runs, len(classes)))
for run in range(runs):

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

        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size)

        #print("All train data shape X_train:", X_train.shape, "Y_train:", Y_train.shape)
        #print("All test data shape X_test:", X_test.shape, "Y_test:", Y_test.shape)

        model = SVC(probability=True, C=C, kernel=kernel)
        model.fit(X_train, Y_train)

        score = model.score(X_train, Y_train)
        #print("Training accuracy:", score)
        score = model.score(X_test, Y_test)
        #print("Test accuracy:", score)
        Y_predict = model.predict_proba(X_test)

        Y_predict = Y_predict[:, 1]
        mAP = average_precision_score(Y_test, Y_predict)
        #print("mAP", mAP)

        best_score = accs.max(axis=0)[i]
        accs[run][i] = score
        mAPs[run][i] = mAP

        # Only save best model
        if score > best_score: 
            new_model_path = model_path.replace(".svm", "_" + str(i) + ".svm")
            #print("Saving model for class", i, "at", new_model_path)
            out_f = open(new_model_path, 'wb')
            pickle.dump(model, out_f)

print("acc\n", accs)
best_accs = accs.max(axis=0)
avg_accs = accs.mean(axis=0)
print("best accuracy", best_accs)
print("avg accuracy", avg_accs)
print("average test accuracy", avg_accs.mean())
print("mAPs\n", mAPs)
best_mAP = mAPs.max(axis=0)
avg_mAP = mAPs.mean(axis=0)
print("best mAP", best_mAP)
print("avg mAP", avg_mAP)
print("average mAP", avg_mAP.mean())
print("C:", C, "| kernel:", kernel, "| test_size:", test_size, "| runs:", runs)
#print("Final test accuracy scores:")
#for i in range(best_scores.shape[0]):
#    print)
