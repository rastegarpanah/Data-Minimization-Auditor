import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.feature_selection import RFE
import os
import pickle
import DM as DM

def save_object(A, filename):
    with open(filename, 'wb') as output:
        pickle.dump(A, output, pickle.DEFAULT_PROTOCOL)

dataset = 'Digits'
model = 'SVM'
n_features = 9

df = pd.read_csv('Data/Datasets/Digits/optdigits.tra',header=None)
y = df[df.columns[-1]].values
data = df[df.columns[0:-1]]
data = data.rename(columns=dict([(f,'f%d'%(f+1)) for f in df.columns]))

data_train, data_test, y_train, y_test = train_test_split(
data, y, test_size=0.5, random_state=0, stratify=y)

clf = LinearSVC(C=0.0001,random_state=0)
#clf = SVC(kernel='rbf', C=500)

### feature selection ###
selector = RFE(clf, n_features_to_select = n_features, step=1)
selector = selector.fit(data_train, y_train)
data_train = data_train[data_train.columns[selector.support_]]
data_test = data_test[data_test.columns[selector.support_]]
data = data[data.columns[selector.support_]]


clf.fit(data_train, y_train)
train_accuracy =  accuracy_score(y_train, clf.predict(data_train))
print("training accuracy:", train_accuracy)
test_accuracy =  accuracy_score(y_test, clf.predict(data_test))
print("test accuracy:", test_accuracy)
print()

R = {'model':clf , 'data':data}
output_path = 'Data/Models/%s-%s'%(dataset, model)
if not os.path.exists(output_path):
    os.makedirs(output_path)
save_object(R, '%s/%d_features'%(output_path,data.shape[1]))
