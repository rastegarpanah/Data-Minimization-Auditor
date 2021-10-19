import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
import os
import pickle
import DM as DM

def save_object(A, filename):
    with open(filename, 'wb') as output:
        pickle.dump(A, output, pickle.DEFAULT_PROTOCOL)
        
dataset = 'Nursery'
model = 'DT'

feature_names = [ 'parents','has-nurs','form','children','housing',
                  'finance','social','health','label']
df = pd.read_csv('Data/Datasets/Nursery/nursery.data', header=None, names=feature_names)
df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x).replace('?', np.nan)
df = df.dropna(axis=0)

y = df['label']
data = df.drop('label',axis=1)

for col in data.select_dtypes(include=[np.object]).columns:
    data[col] = data[col].astype('category')
data_dummies = pd.get_dummies(data)

data_train_dummies, data_test_dummies, y_train, y_test = train_test_split(
data_dummies, y, test_size=0.8, random_state=0, stratify=y)

clf = DecisionTreeClassifier(min_impurity_decrease=0.0001,random_state=1)
clf.fit(data_train_dummies, y_train)
train_accuracy =  accuracy_score(y_train, clf.predict(data_train_dummies))
print("training accuracy:", train_accuracy)
test_accuracy =  accuracy_score(y_test, clf.predict(data_test_dummies))
print("test accuracy:", test_accuracy)
print()

R = {'model':clf , 'data':data}
output_path = 'Data/Models/%s-%s'%(dataset, model)
if not os.path.exists(output_path):
    os.makedirs(output_path)
save_object(R, '%s/%d_features'%(output_path,data.shape[1]))
