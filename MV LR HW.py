import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import root_mean_squared_error

cardf = pd.read_csv ('/Users/arjunbubbar/Desktop/Jetlearn/Data Science/Datasets/CarPrices.csv')

print (cardf.info ())

# Checked for missing values, nothing missing so no need to pre-process

x = cardf.drop (columns = 'price')
y = cardf ['price']

# Selects columns based on data types

nonnumericcol = x.select_dtypes (include = ['object']).columns 

# Encoding required on these columns, label encoder since don't want to have too many columns 

from sklearn.preprocessing import LabelEncoder

for column in nonnumericcol:
    le = LabelEncoder ()
    x[column] = le.fit_transform (x[column])

x.info ()

xtrain, xtest, ytrain, ytest = train_test_split (x,y, test_size=0.2, random_state=40)

model = LinearRegression ()
model.fit (xtrain,ytrain)

print ('Slope', model.coef_)
print ('Intercept', model.intercept_)

predictedvalues = model.predict (xtrain)
predictedvalues2 = model.predict (xtest)

print (root_mean_squared_error (ytrain, predictedvalues))
print (root_mean_squared_error (ytest, predictedvalues2))

from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import root_mean_squared_error

poly = PolynomialFeatures (degree = 20)

polyfeatures = poly.fit_transform (xtrain)
polyfeatures2 = poly.fit_transform (xtest)

model2 = LinearRegression ()
model2.fit (polyfeatures,ytrain)

predictedvalues3 = model2.predict (polyfeatures)
predictedvalues4 = model2.predict (polyfeatures2)

print (root_mean_squared_error (ytrain, predictedvalues3))
print (root_mean_squared_error (ytest, predictedvalues4))

