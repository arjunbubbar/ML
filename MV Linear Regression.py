# Multiple features one target (many x)
# m1x1 + m2x2 + m3x3 + ... + c

import pandas as pd 
from sklearn.impute import SimpleImputer

housingdf = pd.read_csv ('/Users/arjunbubbar/Desktop/Jetlearn/Data Science/Datasets/HousingData.csv')

housingdf.info ()

housingdf = housingdf [['AGE','LSTAT','RM','MEDV']]

impute = SimpleImputer (missing_values=pd.NA, strategy='median')

housingdf [['AGE','LSTAT']] = impute.fit_transform (housingdf [['AGE','LSTAT']])

print (housingdf.isna ().sum ())

x = housingdf [['AGE','LSTAT','RM']]
y = housingdf ['MEDV']

# Split data into training and test set, 70/80% and 20% respectively
# random state ensures that same results go for training and testing, when model enhanced then it should look at the same textbook
# 40 is a random combination of data
# x and y split into train and test, so have 4 datasets and 4 variables, xtrain, xtest, ytrain, ytest
# Predict y value which is MEDV using three gradients + y intercept

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import root_mean_squared_error

xtrain, xtest, ytrain, ytest = train_test_split (x,y, test_size=0.2, random_state=40)

model = LinearRegression ()
model.fit (xtrain,ytrain)

print ('Slope', model.coef_)
print ('Intercept', model.intercept_)

predictedvalues = model.predict (xtrain)
predictedvalues2 = model.predict (xtest)

print (root_mean_squared_error (ytrain, predictedvalues))
print (root_mean_squared_error (ytest, predictedvalues2))









