# -*- coding: utf-8 -*-
"""
Created on Mon Mar 25 15:01:59 2019

@author: Sethu
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

ds=pd.read_csv('Credit_card.csv')
x=ds.iloc[:,1:-1].values
y=ds.iloc[:,24].values

from sklearn.preprocessing import OneHotEncoder
onehot=OneHotEncoder(categorical_features=[1,2,3])
x=onehot.fit_transform(x).toarray()

#avoiding dummy variables, so we remove one column from each categorical variables
x=np.delete(x,[1,9,12],axis=1)

#splitting the dataset into test and train
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.20,random_state=0)

#feature scaling
from sklearn.preprocessing import StandardScaler
sc_x=StandardScaler()
x_train=sc_x.fit_transform(x_train)
x_test=sc_x.fit_transform(x_test)

#Logistic Regression
from sklearn.linear_model import LogisticRegression
log_reg=LogisticRegression()
log_reg.fit(x_train,y_train)
log_pred=log_reg.predict(x_test)

#Random Forest
from sklearn.ensemble import RandomForestClassifier
random_forest=RandomForestClassifier(n_estimators=100)
random_forest.fit(x_train,y_train)
forest_pred=random_forest.predict(x_test)

#KNN
from sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier(n_neighbors=5)
knn.fit(x_train,y_train)
knn_pred=knn.predict(x_test)

#Decison Tree
from sklearn.tree import DecisionTreeClassifier
tree=DecisionTreeClassifier(max_depth=5)
tree.fit(x_train,y_train)
tree_pred=tree.predict(x_test)


#Keras 
import keras
from keras.models import Sequential
from keras.layers import Dense

classifier=Sequential()

classifier.add(Dense(units = 15, kernel_initializer = 'uniform', activation = 'relu', input_dim = 30))
classifier.add(Dense(units = 15, kernel_initializer = 'uniform', activation = 'relu'))
classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
classifier.fit(x_train, y_train, batch_size = 10, epochs = 100)

ann_pred = classifier.predict(x_test)

for index,pred in enumerate(ann_pred):
    if pred<0.5:
        ann_pred[index] = 0
    else:
        ann_pred[index] = 1

#confusion matrix
from sklearn.metrics import confusion_matrix
log_cm=confusion_matrix(y_test,log_pred)
forest_cm=confusion_matrix(y_test,forest_pred)
knn_cm=confusion_matrix(y_test,knn_pred)
tree_cm=confusion_matrix(y_test,tree_pred)
ann_cm = confusion_matrix(y_test, ann_pred)

#validation of the models
from sklearn.model_selection import cross_val_score
print('Accuracy for Logisitic Regression',((log_cm[0,0]+log_cm[1,1])/len(x_test)))
print('Accuracy for Random forest',((forest_cm[0,0]+forest_cm[1,1])/len(x_test)))
print('Accuracy for Knn',((knn_cm[0,0]+knn_cm[1,1])/len(x_test)))
print('Accuracy for Decision Tree',((tree_cm[0,0]+tree_cm[1,1])/len(x_test)))
print('Accuracy for ANN',((ann_cm[0,0]+ann_cm[1,1])/len(x_test)))





