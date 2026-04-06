# pre-processing technique - for dimensionality reduction
# curse of dimensionality - more complex to find a solution as dimensions increase
# too many features lead to complex and time consuming dimensions, can cause overfitting
# overfitting - model has understood too much, cannot learn something new, too focused learning
# therefore reduce number of features, dimensionality reduction, combine the features instead of losing 

# scaling is an important step in pca

import pandas as pd
from sklearn import datasets
from sklearn.preprocessing import MinMaxScaler

bc = datasets.load_breast_cancer ()

print (bc.keys ())

x = pd.DataFrame (bc.data, columns=bc.feature_names)

print (x)

y = pd.Series (bc.target)

print (y)

print (x.info ())

# puts between 0 and 1

mms = MinMaxScaler ()

x = mms.fit_transform (x)

print (x)

# we have 30 features, so pca is useful, n components, select how many to have after existing features have been combined
# combining features makes it even better

from sklearn.decomposition import PCA

pca = PCA (n_components = 10)

x = pca.fit_transform (x)

print (x)

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.svm import SVC

xtrain, xtest, ytrain, ytest = train_test_split (x,y,test_size=0.2,random_state=20)

svc = SVC (kernel='linear')

svc.fit (xtrain, ytrain)

predy = svc.predict (xtest)

print (predy)

print (classification_report (ytest, predy))
print (confusion_matrix (ytest, predy))
print (f1_score (ytest, predy))

