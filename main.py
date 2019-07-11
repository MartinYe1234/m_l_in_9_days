"""
Random forest on grocery store data. 
Aiming for 80% accuracy and plus
"""

#import necessary modules
from sklearn.ensemble import RandomForestRegressor
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from scipy.io import arff
import pandas as pd
import os
"""
Organizing and formatting data
"""
#access to data
os.chdir(r"C:\Users\User\Desktop")
storeData = pd.read_csv('Sales_Transactions_Dataset_Weekly.csv')

#format data
df = pd.DataFrame(storeData)
#access only the data that is to be used
xVar = df.drop('Normalized 51',axis = 1)
for i in range(52):
    xVar = xVar.drop(('W'+str(i)),axis = 1)
xVar = xVar.drop('MIN',axis = 1)
xVar = xVar.drop('MAX',axis = 1)
xVar = xVar.drop('Product_Code',axis = 1)
yVar = df['Normalized 51']

#splits data into 80% training and 20% testing
X_train, X_test, y_train, y_test = train_test_split(xVar, yVar, test_size=0.2,shuffle = False)

"""
Random Forest
"""
rnd_clf = RandomForestRegressor(n_estimators=500, max_leaf_nodes=16, n_jobs=-1)
rnd_clf.fit(X_train, y_train)

y_pred_rf = rnd_clf.predict(X_test)
print(y_pred_rf)
print(yVar)