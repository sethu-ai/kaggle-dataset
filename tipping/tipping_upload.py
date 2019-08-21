# -*- coding: utf-8 -*-
"""
Created on Thu Mar 28 12:01:53 2019

@author: victor
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

ds=pd.read_csv('tips.csv')
x=ds.iloc[:,[0,2,3,4,5,6]].values
y=ds.iloc[:,[1]].values

null=ds.isna().sum()

from sklearn.preprocessing import LabelEncoder,OneHotEncoder
label_one=LabelEncoder()
label_two=LabelEncoder()
label_three=LabelEncoder()
label_four=LabelEncoder()

x[:,1]=label_one.fit_transform(x[:,1])
x[:,2]=label_one.fit_transform(x[:,2])
x[:,3]=label_one.fit_transform(x[:,3])
x[:,4]=label_one.fit_transform(x[:,4])

hot=OneHotEncoder(categorical_features=[3])
x=hot.fit_transform(x).toarray()

#removing dummy variable
x=x[:,[0,1,2,4,5,6,7,8]]

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)


from sklearn.linear_model import LinearRegression
linear=LinearRegression()
linear.fit(x_train,y_train)
linear_pred=linear.predict(x_test)

from sklearn import metrics
linear_accuracy=metrics.mean_squared_error(y_test,linear_pred)

print("Accuracy for linear regression:",linear_accuracy*100)

a=x_test[:,3]
plt.scatter(a,y_test,color='blue')
plt.scatter(a,linear_pred,color='red')
plt.show()