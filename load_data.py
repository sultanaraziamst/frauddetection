import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import gc
import os
from IPython.display import Markdown, display

warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', 100)

train_transaction = pd.read_csv('C:\\Users\\LENOVO\\Desktop\\Project with Pujan\\frauddetection\\data\\train_transaction..csv')
print(train_transaction.head())
test_transaction = pd.read_csv('C:\\Users\\LENOVO\Desktop\\Project with Pujan\\frauddetection\\data\\test_transaction.csv')


#Load identity data
print("Loading identity data...")
train_identity = pd.read_csv('C:\\Users\\LENOVO\\Desktop\\Project with Pujan\\frauddetection\\data\\train_identity.csv')
test_identity = pd.read_csv('C:\\Users\\LENOVO\Desktop\\Project with Pujan\\frauddetection\\data\\test_identity.csv')


