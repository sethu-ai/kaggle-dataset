# -*- coding: utf-8 -*-
"""
Created on Thu Jul  4 15:09:52 2019

@author: Sethu
"""

#importing necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

#reading in the dataset 
df=pd.read_csv('heart.csv')

#after visualising the dataset we infer that the dataset needs to be shuffled because the dependent variables are polarized.

shuffled_df=df.sample(frac=1)

# alternative option for shuffling
'''from sklearn.utils import shuffle
df=shuffle(df)'''

x=shuffled_df.iloc[:,:-1].values
y=shuffled_df.iloc[:,13].values

from sklearn.preprocessing import OneHotEncoder
onehot=OneHotEncoder(categorical_features=[2,6,10,11,12])
x=onehot.fit_transform(x).toarray()

x=np.delete(x,[3,6,9,14,18],axis=1)

#feature scaling
from sklearn.preprocessing import StandardScaler
sc_x=StandardScaler()
x=sc_x.fit_transform(x)

#Splitting the dataset into train and test
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.20,random_state=0)

#Logistic Regression
from sklearn.linear_model import LogisticRegression
log_reg=LogisticRegression()
log_reg.fit(x_train,y_train)
log_pred=log_reg.predict(x_test)

#Random Forest
from sklearn.ensemble import RandomForestClassifier
forest=RandomForestClassifier()
forest.fit(x_train,y_train)
forest_pred=forest.predict(x_test)

#KNN
from sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier()
knn.fit(x_train,y_train)
knn_pred=knn.predict(x_test)

#Decision Tree
from sklearn.tree import DecisionTreeClassifier
tree=DecisionTreeClassifier()
tree.fit(x_train,y_train)
tree_pred=tree.predict(x_test)

'''#XG Boost
from XGBoost import XGBclassifier
xg=XGBclassifier()
xg.fit(x_train,y_train)
xg_pred=xg.predict(x_test)'''


#confusion matrix
from sklearn.metrics import confusion_matrix
log_cm=confusion_matrix(y_test,log_pred)  
forest_cm=confusion_matrix(y_test,forest_pred)
knn_cm=confusion_matrix(y_test,knn_pred)
tree_cm=confusion_matrix(y_test,tree_pred)
#xg_cm=confusion_matrix(y_test,xg_pred)


#Accuracy 
print("Accuracy for the logistic Regression",(log_cm[0,0]+log_cm[1,1])/len(x_test))                              
print("Accuracy for the Random Forest",(forest_cm[0,0]+forest_cm[1,1])/len(x_test))         
print("Accuracy for the KNN", (knn_cm[0,0]+knn_cm[1,1])/len(x_test))         
print("Accuracy for the Decision Tree", (tree_cm[0,0]+tree_cm[1,1])/len(x_test))         
#print("Accuracy for the XGBoost", (xg_cm[0,0]+xg_cm[1,1])/len(x_test))         
