# Regression in Python ----

# Linear ----

# Classification ----
import sklearn
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = iris.target
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)
from sklearn.linear_model import LogisticRegression
model = LogisticRegression(solver='lbfgs',
multi_class ='multinomial')
model.fit(X_train,y_train)
from sklearn.metrics import confusion_matrix
# Test result
y_predict = model.predict(X_test)
confusion_matrix(y_test, y_predict)
from sklearn.metrics import accuracy_score
accuracy_score(y_test, y_predict)*100
