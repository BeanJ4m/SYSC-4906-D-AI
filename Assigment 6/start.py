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
