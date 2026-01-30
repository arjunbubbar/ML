
# Model predicts polynomial, version of linear regressions
# When the relationship is non-linear, the linear regression will not give good results
# y = m1x1^2 + m2x2^2 etc.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer

# Example 

r = np.arange (1,25)

area = np.pi * r**2

plt.plot (r,area)

plt.show ()

# Housing example

housingdf = pd.read_csv ('/Users/arjunbubbar/Desktop/Jetlearn/Data Science/Datasets/HousingData.csv')

housingdf = housingdf [['LSTAT','RM','MEDV']]

impute = SimpleImputer (missing_values=pd.NA, strategy='median')

housingdf [['LSTAT']] = impute.fit_transform (housingdf [['LSTAT']])

print (housingdf.isna ().sum ())

x = housingdf [['LSTAT','RM']]
y = housingdf ['MEDV']

print (x)

# Transforming features to higher degree, so it would do LSTAT^2, LSTAT*RM, RM^2 for example if degree = 2

from sklearn.preprocessing import PolynomialFeatures

poly = PolynomialFeatures (degree = 2)

polyfeatures = poly.fit_transform (x)
print (polyfeatures)

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import root_mean_squared_error

xtrain, xtest, ytrain, ytest = train_test_split (polyfeatures,y, test_size=0.2, random_state=40)

model = LinearRegression ()
model.fit (xtrain,ytrain)

print ('Slope', model.coef_)
print ('Intercept', model.intercept_)

predictedvalues = model.predict (xtrain)
predictedvalues2 = model.predict (xtest)

print (root_mean_squared_error (ytrain, predictedvalues))
print (root_mean_squared_error (ytest, predictedvalues2))





