import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import root_mean_squared_error

salarydf = pd.read_csv ('/Users/arjunbubbar/Desktop/Jetlearn/Data Science/Datasets/Salary.csv')

x = salarydf ['YearsExperience']
y = salarydf ['Salary']

plt.scatter (x,y)
plt.xlabel("YearsExperience")
plt.ylabel("Salary")
plt.title("Salary vs Years of Experience")
plt.show()

xreshaped = x.values.reshape (-1,1)
model = LinearRegression ()
model.fit (xreshaped,y)

predictedvalues = model.predict (xreshaped)

print ('Slope', model.coef_[0])
print ('Intercept', model.intercept_)
print (root_mean_squared_error (y, predictedvalues))

while True:
    user = input("Enter years of experience (or type quit): ")

    if user == "quit":
        print("Bye")
        break

    years = float (user)
    pred_salary = model.predict([[years]])[0]
    print(f"Predicted salary for {years} years experience: {pred_salary:.2f}")



