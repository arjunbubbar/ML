# 3 types of ML - Supervised, Unsupervised, Reinforced (feedback) 

# 3 types of problem solving - Classification, Regression (predict value), Recommendation (based on user historical data)

import matplotlib.pyplot as plt

import numpy as np

x = np.arange (1,11)

y = np.array ([23,26,27,34,38,39,45,47,48,50])

plt.scatter (x,y)

plt.show ()

#find line of best fit
#m = sum((xi-mean(x)) * (yi-mean(y))) / sum((xi – mean(x))^2)
#c = mean(y) – m * mean(x)

meanx = np.mean (x)

meany = np.mean (y)

m = np.sum ((x-meanx)*(y-meany))/np.sum ((x-meanx)**2)

c = meany - m * meanx

print ('Slope', m)
print ('Intercept', c)

predictedy = m * x + c
print (predictedy)

plt.scatter (x,y)

plt.plot (x,predictedy)

plt.show ()


# Evaluating the model - find difference between actual and predicted values 

# RMSE - Root Mean Squared Error sqrt( sum( (p – yi)^2 )/n 


rmse = np.sqrt (np.mean ((predictedy - y)**2))

print (rmse)


from sklearn.linear_model import LinearRegression

# many features (x), one target (y)

# transform 1-D data into 2-D table (-1 unspecified, 1 column)

xreshaped = x.reshape (-1, 1)

# Create object, train model, use model to predict

model = LinearRegression ()

model.fit (xreshaped,y)

predictedvalues = model.predict (xreshaped)

print ('Slope', model.coef_)
print ('Intercept', model.intercept_)
print (predictedvalues)


