# classification and regression
# decision boundary, like a hyper plane in an n-dimensional surface
# features 2 then hyperplane a line, features 3 then hyperplane two dimensional plane
# find a plane with maximum distance betwen datapoints in classes - margin
# support vectors influence the position of the hyperplane to max margin
# if you cannot separate data with straight line use non-linear SVM, then increases dimension and finds a hyperplane
# kernel is a parameter to obtain an optimum decision surface
# linear kernel creates hyperplane, kernel to rbf or poly they are non-linear hyperplanes

from sklearn import datasets
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, f1_score

bc = datasets.load_breast_cancer ()

print (bc.keys ())

x = pd.DataFrame (bc.data, columns=bc.feature_names)

print (x)

y = pd.Series (bc.target)

print (y)

print (x.info ())

xtrain, xtest, ytrain, ytest = train_test_split (x,y,test_size=0.2,random_state=20)

svc = SVC (kernel='linear')

svc.fit (xtrain, ytrain)

predy = svc.predict (xtest)

print (predy)

print (classification_report (ytest, predy))
print (confusion_matrix (ytest, predy))
print (f1_score (ytest, predy))


