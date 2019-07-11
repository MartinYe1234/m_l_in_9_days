"""
Random forest on grocery store data. 
Aiming for 80% accuracy and plus
"""

#import necessary modules
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from sklearn.model_selection import train_test_split
from scipy.io import arff
import pandas as pd
import os

#access to data
os.chdir(r"C:\Users\User\Desktop")
storeData = pd.read_csv('Sales_Transactions_Dataset_Weekly.csv')
#format data
df = pd.DataFrame(storeData)
#x_train is the training from column 0 to 50 - THIS NEEDS TO BE A LIST
xVar = df.loc[:,'Normalized 0':'Normalized 50']

print(xVar)
#y_train is the test - column 51
yVar = df['Normalized 51']   
#splits data into 80% training and 20% testing
X_train, X_test, y_train, y_test = train_test_split(xVar, yVar, test_size=0.2)

clf = RandomForestClassifier(n_jobs=2, random_state=0)

clf.fit(X_train, y_train)

RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
            max_depth=None, max_features='auto', max_leaf_nodes=None,
            min_impurity_split=1e-07, min_samples_leaf=1,
            min_samples_split=2, min_weight_fraction_leaf=0.0,
            n_estimators=10, n_jobs=2, oob_score=False, random_state=0,
            verbose=0, warm_start=False)

predictions = clf.predict(X_test)
#check accuracy
pd.crosstab(y_test, predictions, rownames=['Actual Result'], colnames=['Predicted Result'])
