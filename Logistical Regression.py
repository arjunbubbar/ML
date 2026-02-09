# Classification using best fit line into the sigmoid function, decides the threshold and passes value as 0 or 1 
# Whether passengers survived or not

import pandas as pd 
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

titanicdf = pd.read_csv ('/Users/arjunbubbar/Desktop/Jetlearn/Data Science/Datasets/titanic.csv')

titanicdf.info ()

le = LabelEncoder ()

titanicdf ['Sex'] = le.fit_transform (titanicdf ['Sex'])

x = titanicdf [['Pclass', 'Age','Siblings/Spouses Aboard','Parents/Children Aboard','Sex']]
y = titanicdf ['Survived']

xtrain, xtest, ytrain, ytest = train_test_split (x,y, test_size=0.2, random_state=40)

model = LogisticRegression ()

model.fit (xtrain, ytrain)

predy = model.predict (xtest)

print (predy)

# To evaluate the model you cannot use rmse, for classification
# Confusion matrix - front diagonal gives accurate values based on the actual and predicted values, reverse diagonal gives incorrect predictions
# Classification report classifies 1 and 0 as positive and negative respectively
# accuracy = TP + TN / TP + TN + FP + FN, precision = TP / TP + FP
# recall = TP / TP + FN (out of the actual class 1 how many was the model able to predict)
# better the model the more close to 1 the value is 
# f1 score gives a score of the model, calculated out of precision and recall, balancing and calculating a value

cm = confusion_matrix (ytest, predy)

print (cm)

cr = classification_report (ytest, predy)

print (cr)

# seaborn is extra visualisation built around pyplot, more options, darker colours better, shows numbers with annot=true

sns.heatmap (cm, annot=True)

plt.show ()




