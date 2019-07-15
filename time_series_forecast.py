"""
Time series forecast to predict grocery sales.
"""
#import statements
import os
import pandas as pd
import numpy as np
import matplotlib.pylab as plt
from matplotlib.pylab import rcParams

#read data and store in data
df = pd.read_csv('Sales_Transactions_Dataset_Weekly.csv')
"""
Plotting
"""
#y = df.loc[:,'Normalized 0':'Normalized 50']
y = df['']