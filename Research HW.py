import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import root_mean_squared_error
from sklearn.preprocessing import PolynomialFeatures

from sklearn.svm import SVR

btc_df = pd.read_csv("/Users/arjunbubbar/Desktop/Jetlearn/Data Science/Datasets/btcusd_1-min_data.csv")

print(btc_df.info())

btc_df = btc_df[['Open','High','Low','Volume','Close']]

impute = SimpleImputer(missing_values=pd.NA, strategy='median')

btc_df[['Open','High','Low','Volume']] = impute.fit_transform (btc_df[['Open','High','Low','Volume']])

print(btc_df.isna().sum())

btc_df = btc_df.sample(n=1000, random_state=40)

x = btc_df[['Open','High','Low','Volume']]
y = btc_df['Close']

xtrain, xtest, ytrain, ytest = train_test_split(x,y,test_size=0.2,random_state=40)

model = LinearRegression()
model.fit(xtrain,ytrain)

print('Slope', model.coef_)
print('Intercept', model.intercept_)

predictedvalues = model.predict(xtrain)
predictedvalues2 = model.predict(xtest)

print(root_mean_squared_error(ytrain, predictedvalues))
print(root_mean_squared_error(ytest, predictedvalues2))

poly = PolynomialFeatures(degree = 5)

polyfeatures = poly.fit_transform(xtrain)
polyfeatures2 = poly.fit_transform(xtest)

model2 = LinearRegression()
model2.fit(polyfeatures,ytrain)

predictedvalues3 = model2.predict(polyfeatures)
predictedvalues4 = model2.predict(polyfeatures2)

