import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
import os
import random
import pickle
import DM as DM

def save_object(A, filename):
    with open(filename, 'wb') as output:
        pickle.dump(A, output, pickle.DEFAULT_PROTOCOL)
        
dataset = 'Census'
model = 'DT'

feature_names=['age','workclass','fnlwgt','education','education-num',
               'marital-status','occupation','relationship','race',
               'sex','capital-gain','capital-loss','hours-per-week',
               'native-country','label']

df = pd.read_csv('Data/Datasets/Census/adult.data', header=None, names=feature_names)
df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x).replace('?', np.nan)        
df = df.dropna(axis=0)
df = df.replace('<=50K', 0)
df = df.replace('>50K',  1)
df = df.drop(['fnlwgt','capital-gain','capital-loss'],axis=1)
y = df['label'].astype(int)
data = df.drop('label',axis=1)

for col in data.select_dtypes(include=[np.object]).columns:
    data[col] = data[col].astype('category')
data_dummies = pd.get_dummies(data)

data_train_dummies, data_test_dummies, y_train, y_test = train_test_split(
data_dummies, y, test_size=0.8, random_state=0, stratify=y)

### Train ###
clf = DecisionTreeClassifier(min_impurity_decrease=0.0001,random_state=1)
clf.fit(data_train_dummies, y_train)
train_accuracy =  accuracy_score(y_train, clf.predict(data_train_dummies))
print("training accuracy:", train_accuracy)
test_accuracy =  accuracy_score(y_test, clf.predict(data_test_dummies))
print("test accuracy:", test_accuracy)
print()

### Feature Selection ###
k=5
importances = DM.feature_importance_categorical(clf,data_dummies.columns)
topk = importances.groupby(level=0).sum().sort_values(ascending=False)[0:k].index.tolist()

data_k = data[topk]
data_dummies_k = pd.get_dummies(data_k)
data_train_dummies_k, data_test_dummies_k, y_train, y_test = train_test_split(
data_dummies_k, y, test_size=0.8, random_state=0, stratify=y)

### Retrain ###
clf.fit(data_train_dummies_k, y_train)
train_accuracy =  accuracy_score(y_train, clf.predict(data_train_dummies_k))
print("training accuracy:", train_accuracy)
test_accuracy =  accuracy_score(y_test, clf.predict(data_test_dummies_k))
print("test accuracy:", test_accuracy)
print()

### Build a new classifier by embedding clf as its prediction model
clf_extra_input = DM.RedundantModel(clf, data_dummies_k.columns.tolist())

### Add extra feature
extra_feature =  ['relationship']
data = data[topk + extra_feature]

output_path = 'Data/Models/%s-%s'%(dataset, model)
if not os.path.exists(output_path):
    os.makedirs(output_path)
R = {'model':clf_extra_input , 'data':data}
save_object(R, '%s/%d_features-redundant'%(output_path,data.shape[1]))
