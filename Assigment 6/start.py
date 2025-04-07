# Import libraries
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
# Load the dataset from a CSV file
df = pd.read_csv("Assigment 6/Eight_Class_Dataset.csv")
# Display the first 5 rows to get a glimpse of the data
print(df.head())
# Check the shape of the DataFrame ( rows , columns )
print("Dataset shape: ", df.shape)
# List the column names
print("Columns: ", df.columns.tolist())
# Check the class distribution in the target column
print(df['Attack_type'].value_counts())

# Shuffle the DataFrame rows
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# Remove rows with any NaN ( missing ) values
df = df.dropna()

print(df['Attack_type'].value_counts())
Label_names = ['Normal', 'DDoS_UDP', 'DDoS_ICMP', 'DDoS_TCP', 'DDoS_HTTP',
               'Password', 'Vulnerability_scanner', 'Uploading', 'SQL_injection']
Label_numbers = [0, 1, 2, 3, 4, 5, 6, 7, 8]
mapping = dict(zip(Label_names, Label_numbers))
df['Attack_type'] = df['Attack_type'].map(mapping)
print(df['Attack_type'].value_counts())

# One - hot encode all categorical feature columns
# ( Assume ’ CategoryFeature ’ is a categorical column in the dataset )
df=pd.get_dummies(df,columns=['CategoryFeature'])
df=df.sample(frac=1,random_state=42).reset_index(drop=True)
print("Columns after one-hot encoding:",df.columns.tolist())