"""
Random forest on grocery store data. 
Aiming for 80% accuracy and plus
"""

#import necessary modules
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
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
# for i in range(52):
#     xVar = xVar.drop(('W'+str(i)),axis = 1)
# xVar = xVar.drop('MIN',axis = 1)
# xVar = xVar.drop('MAX',axis = 1)
# xVar = xVar.drop('Product_Code',axis = 1)
xVar = df.loc[:,'Normalized 0':'Normalized 50']
yVar = df.iloc[:,106]

#used to check if

#splits data into 80% training and 20% testing
X_train, X_test, y_train, y_test = train_test_split(xVar, yVar, test_size=0.2)

"""
Random Forest
"""
rnd_clf = RandomForestRegressor(n_estimators=1000, criterion="mse", n_jobs=-1)

rnd_clf.fit(X_train, y_train)

#make predictions
y_pred = rnd_clf.predict(X_test)
"""
Graphing
"""
#Create rectangle   
fig = plt.figure()
#data to plot
actual = [i for i in range(51)]
predictions = [i+1 for i in range(51)]
#x axis is number of days which is 51 days from day 0 to 50
x_axis = [i for i in range(51)]
#label along y axis
plt.ylabel("Units sold")
#label along x axis
plt.xlabel("Days")
#title
plt.title("Units actually sold and Units predicted")
#plot data
plt.plot(x_axis,actual)
plt.plot(x_axis,predictions)
plt.show()


