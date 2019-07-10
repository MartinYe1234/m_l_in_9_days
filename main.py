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




#split data into test and train
