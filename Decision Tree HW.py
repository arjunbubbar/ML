import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, f1_score

from sklearn.tree import DecisionTreeClassifier

# if the file is not comma separated use sep and then specify 

bank_df = pd.read_csv ('/Users/arjunbubbar/Desktop/Jetlearn/Data Science/Datasets/bank.csv',sep=';')

nonnumericalcols = bank_df.select_dtypes (include = 'object').columns 

le = LabelEncoder ()

# encodes all values in object columns iteratively 

for i in nonnumericalcols:
    bank_df [i] = le.fit_transform (bank_df [i])

print (bank_df.info ())

x = bank_df.drop (columns=['y'])
y = bank_df ['y']

print (y.value_counts ())

# xtrain, xtest, ytrain, ytest = train_test_split (x,y, test_size= 0.2, random_state= 20)
# dtc = DecisionTreeClassifier ()

# dtc.fit (xtrain,ytrain)
# predy = dtc.predict (xtest)
# print (predy)

# print(f1_score(ytest, predy))
# print(confusion_matrix(ytest, predy))
# print(classification_report(ytest, predy))





