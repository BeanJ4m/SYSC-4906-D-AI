# Import libraries
from sklearn . preprocessing import MinMaxScaler
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
# ( Assume ' Attack_type ' is a categorical column in the dataset )
df = pd.get_dummies(df, columns=['Attack_type'])
df = df.sample(frac=1, random_state=42).reset_index(drop=True)
print("Columns after one-hot encoding:", df.columns.tolist())


# Separate features and labels
X = df.drop('Attack_type_1', axis=1)
y = df['Attack_type_1']
# all feature columns
# the numeric labels
# First split : Training + Validation vs Test
X_train_val, X_test, y_train_val, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)
# Second split : Training vs Validation ( split the 80% further into 60/20)
X_train, X_val, y_train, y_val = train_test_split(
    X_train_val, y_train_val, test_size=0.25, random_state=42, stratify=y_train_val)
print(X_test.shape)
print(y_test.shape)
print(X_train.shape)
print(y_train.shape)
print(X_val.shape)
print(y_val.shape)


# Initialize the scaler
scaler = MinMaxScaler()
# Fit on training features , then transform train , val , and test features
X_train_scaled = scaler . fit_transform(X_train)
X_val_scaled = scaler . transform(X_val)
X_test_scaled = scaler . transform(X_test)


class ClassifierDataset (Dataset):
    def __init__(self, features, labels):
        # features : numpy array or pandas DataFrame of shape (N , d )
        # labels : array or Series of length N
        self.X = features.astype('float32')
        self.y = labels.astype('int64')

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        # get features and label at index idx
        return torch.tensor(self.X[idx]), torch.tensor(self.y[idx])


# Create Dataset instances for train and validation sets
train_dataset = ClassifierDataset(X_train_scaled, y_train . values)
val_dataset = ClassifierDataset(X_val_scaled, y_val . values)
test_dataset = ClassifierDataset(X_test_scaled, y_test . values)
# Create DataLoaders to load data in batches
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)


# Gradient Boosting Classifier with updated labels
Label_names = [' Normal ', ' DDoS_UDP ', ' DDoS_ICMP ', ' DDoS_TCP ', ' DDoS_HTTP ',
               'Password ', ' Vul nerab ilit y_sca nner ', ' Uploading ', ' SQL_injection ']
Label_numbers = [0, 1, 2, 3, 4, 5, 6, 7, 8]

gb_model = GradientBoostingClassifier(
    n_estimators=20, learning_rate=0.1, random_state=42)
gb_model.fit(X_train_scaled, y_train)
y_test_pred_gb = gb_model.predict(X_test_scaled)
print("GradientBoostingClassificationReport:")
print(classification_report(y_test, y_test_pred_gb,
      labels=Label_numbers, target_names=Label_names))
# Confusionmatrix-GB(counts)
cm_gb = confusion_matrix(y_test, y_test_pred_gb, labels=Label_numbers)
sns.heatmap(cm_gb, annot=True, fmt='d', cmap='gray',
            xticklabels=Label_names, yticklabels=Label_names)
plt.title("GradientBoostingConfusionMatrix(Counts)")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()
# Confusionmatrix-GB(percentages)
cm_gb_percent = cm_gb.astype('float')/cm_gb.sum(axis=1)[:, np.newaxis]*100
sns.heatmap(cm_gb_percent, annot=True, fmt='.2f', cmap='gray',
            xticklabels=Label_names, yticklabels=Label_names)
plt.title("GradientBoostingConfusionMatrix(Percentages)")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()


class SimpleDNN ( nn . Module ) :
    def __init__ ( self , input_size , num_classes ) :
        super ( SimpleDNN , self ) . __init__ ()
        self . fc1 = nn . Linear ( input_size , 32)
        # first layer ( input -> hidden )
        self . relu = nn . ReLU ()
        # ReLU activation
        self . fc2 = nn . Linear (32 , num_classes ) # second layer ( hidden -> output )
        
    def forward ( self , x ) :
        x = self . fc1 ( x )
        x = self . relu ( x )
        x = self . fc2 ( x )
        # raw scores for each class
        return x
    
# Initialize the network
unique_classes = 9
model = SimpleDNN ( X_train_scaled . shape [1] , unique_classes )
# Define loss function and optimizer
criterion = nn . CrossEntropyLoss ()
optimizer = torch . optim . Adam ( model . parameters () , lr =0.01)


# Training loop
num_epochs = 20
train_losses , val_losses , train_accs , val_accs , test_accs = [] , [] , [] , [] , []
for epoch in range ( num_epochs ) :
    model . train ()
    total_loss , correct , total = 0 , 0 , 0
    for X_batch , y_batch in train_loader :
        optimizer . zero_grad () # reset gradients
        outputs = model ( X_batch ) # forward pass
        loss = criterion ( outputs , y_batch ) # compute loss
        loss . backward () # backpropagation
        optimizer . step () # update parameters
        total_loss += loss . item () * X_batch . size (0) # accumulate loss
        _ , preds = torch . max ( outputs , 1) # predicted class indices
        correct += ( preds == y_batch ) . sum () . item ()
        total += y_batch . size (0)
    train_losses . append ( total_loss / total )
    train_accs . append ( correct / total )
    model . eval () # evaluation mode ( disables dropout , if any )
    val_loss , val_correct , val_total = 0 , 0 , 0
    with torch . no_grad () :
    # no gradient needed for eval
    
        for X_batch , y_batch in val_loader :
            outputs = model ( X_batch )
            loss = criterion ( outputs , y_batch )
            val_loss += loss . item () * X_batch . size (0)
            _ , preds = torch . max ( outputs , 1)
            val_correct += ( preds == y_batch ) . sum () . item ()
            val_total += y_batch . size (0)
    val_losses . append ( val_loss / val_total )
    val_accs . append ( val_correct / val_total )
    print(f"Epoch {epoch}: Train Loss = {total_loss / total:.4f}, Train Acc = {correct / total:.4f}, "
          f"Val Loss = {val_loss / val_total:.4f}, Val Acc = {val_correct / val_total:.4f}")
    # Evaluate on test set each epoch
    test_correct , test_total = 0 , 0
    with torch . no_grad () :
        for X_batch , y_batch in test_loader :
            outputs = model ( X_batch )
            _ , preds = torch . max ( outputs , 1)
            test_correct += ( preds == y_batch ) . sum () . item ()
            test_total += y_batch . size (0)
    test_acc = test_correct / test_total
    test_accs . append ( test_acc )
    

#Plotlossandaccuracy
plt.style.use('ggplot')
plt.plot(train_losses,label='TrainLoss')
plt.plot(val_losses,label='ValidationLoss',linestyle='--')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()
plt.plot(train_accs,label='TrainAccuracy')
plt.plot(val_accs,label='ValidationAccuracy',linestyle='--')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
#Plottestaccuracyoverepochs
plt.plot(test_accs,label='TestAccuracy',marker='o')
plt.xlabel('Epoch')
plt.ylabel('TestAccuracy')
plt.title('TestAccuracyOverEpochs')
plt.legend()
plt.show()

