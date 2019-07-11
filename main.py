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
#x_train is the training - wip
x_train = df["W0"]
#y_train is the test
y_train = df['Normalized 51']   
print(df)
print(x_train)


"""
xVar = df.loc['']

:)

yVar = df.iloc[:,20]
df2 = df[xVar]
"""
#split data into test and train - p1 (first 700 no prediction last 100 50 days train 1 day test)
#x - train 50 days
#y - test 1 day

