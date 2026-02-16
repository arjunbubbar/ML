import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from imblearn.over_sampling import RandomOverSampler

admissionsdf = pd.read_csv ('/Users/arjunbubbar/Desktop/Jetlearn/Data Science/Datasets/admission.csv')

print (admissionsdf.info ())

x = admissionsdf [['gre','gpa','rank']]
y = admissionsdf ['admit']

print (y.value_counts ())

# resampling - undersampling (to cut down data size) or oversampling (to have artificial data implemented)
# use imblearn library, auto resamples all the minority class, balance the data 

ros = RandomOverSampler (sampling_strategy = 'auto')

x,y = ros.fit_resample (x,y)

print (y.value_counts ())

xtrain, xtest, ytrain, ytest = train_test_split (x, y, test_size = 0.1, random_state = 40)

model = LogisticRegression ()

model.fit (xtrain, ytrain)

predy = model.predict (xtest)

print (predy)

cm = confusion_matrix (ytest, predy)

print (cm)

cr = classification_report (ytest, predy)

print (cr)

