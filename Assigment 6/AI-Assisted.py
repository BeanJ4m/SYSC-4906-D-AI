import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# --------------------------
# 1. Dynamic Data Loading & Validation
# --------------------------
def load_and_validate_data(file_path):
    try:
        df = pd.read_csv(file_path)
        
        # Dynamic column detection
        required_prefixes = {
            'target': ['Attack_type'],
            'network_features': ['tcp', 'udp', 'ip', 'dns', 'http', 'mqtt'],
            'security_features': ['malicious', 'scan', 'injection']
        }

        # Validate target column
        target_cols = [col for col in df.columns 
                      if any(pre in col for pre in required_prefixes['target'])]
        if not target_cols:
            raise ValueError("No target column detected")
        target_col = target_cols[0]

        # Validate feature columns
        feature_cols = [col for col in df.columns if col != target_col and
                       any(any(pre in col for pre in pre_list) 
                           for pre_list in required_prefixes.values())]
        
        if not feature_cols:
            raise ValueError("No valid feature columns detected")

        print(f"Detected target column: {target_col}")
        print(f"Detected {len(feature_cols)} feature columns")
        
        return df[[target_col] + feature_cols], target_col, feature_cols

    except Exception as e:
        print(f"Data loading error: {str(e)}")
        exit()

# Load data with dynamic validation
df, target_col, feature_cols = load_and_validate_data("Assigment 6/Eight_Class_Dataset.csv")

# --------------------------
# 2. Dynamic Preprocessing Pipeline
# --------------------------
# Clean data
df = df.dropna(axis=1, how='all')
df = df.dropna()

# Convert numeric columns dynamically
numeric_cols = df[feature_cols].select_dtypes(include=np.number).columns.tolist()
non_numeric = list(set(feature_cols) - set(numeric_cols))

for col in non_numeric:
    try:
        df[col] = pd.to_numeric(df[col], errors='raise')
        numeric_cols.append(col)
    except:
        pass  # Keep as categorical

# Auto-detect categorical columns (non-numeric remaining)
categorical_cols = [col for col in feature_cols 
                   if col not in numeric_cols and col != target_col]

# Dynamic one-hot encoding
if categorical_cols:
    print(f"One-hot encoding {len(categorical_cols)} categorical columns")
    df = pd.get_dummies(df, columns=categorical_cols, dummy_na=True)
else:
    print("No categorical columns detected")

# --------------------------
# 3. Dynamic Label Handling
# --------------------------
# Auto-detect class labels
label_names = sorted(df[target_col].unique().tolist())
label_numbers = list(range(len(label_names)))
label_mapping = dict(zip(label_names, label_numbers))
df[target_col] = df[target_col].map(label_mapping)

# --------------------------
# 4. Data Splitting & Scaling
# --------------------------
X = df.drop(target_col, axis=1)
y = df[target_col]

try:
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=0.25, random_state=42, stratify=y_train_val)
except ValueError as e:
    print(f"Stratification error: {str(e)}")
    exit()

scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# --------------------------
# 5. Dynamic Model Implementation
# --------------------------
# Gradient Boosting Classifier
gb_model = GradientBoostingClassifier(n_estimators=20, learning_rate=0.1, random_state=42)
gb_model.fit(X_train_scaled, y_train)

# PyTorch Dataset
class DynamicDataset(Dataset):
    def __init__(self, features, labels):
        self.X = features.astype('float32')
        self.y = labels.astype('int64')
    
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        return torch.tensor(self.X[idx]), torch.tensor(self.y[idx])

# Create DataLoaders
def create_loaders(train, val, test, batch_size=32):
    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader

train_dataset = DynamicDataset(X_train_scaled, y_train.values)
val_dataset = DynamicDataset(X_val_scaled, y_val.values)
test_dataset = DynamicDataset(X_test_scaled, y_test.values)
train_loader, val_loader, test_loader = create_loaders(train_dataset, val_dataset, test_dataset)

# Dynamic DNN Architecture
class DynamicDNN(nn.Module):
    def __init__(self, input_size, output_size):
        super(DynamicDNN, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, output_size)
        )
    
    def forward(self, x):
        return self.layers(x)

# Initialize model
model = DynamicDNN(X_train_scaled.shape[1], len(label_names))
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# --------------------------
# 6. Training & Evaluation
# --------------------------
def train_model(model, criterion, optimizer, epochs=20):
    metrics = {'train_loss': [], 'val_loss': [], 
              'train_acc': [], 'val_acc': []}
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss, train_correct = 0, 0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * X_batch.size(0)
            _, preds = torch.max(outputs, 1)
            train_correct += (preds == y_batch).sum().item()
        
        # Validation phase
        model.eval()
        val_loss, val_correct = 0, 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                val_loss += loss.item() * X_batch.size(0)
                _, preds = torch.max(outputs, 1)
                val_correct += (preds == y_batch).sum().item()
        
        # Store metrics
        metrics['train_loss'].append(train_loss/len(train_loader.dataset))
        metrics['val_loss'].append(val_loss/len(val_loader.dataset))
        metrics['train_acc'].append(train_correct/len(train_loader.dataset))
        metrics['val_acc'].append(val_correct/len(val_loader.dataset))
        
        print(f"Epoch {epoch+1}/{epochs} | "
              f"Train Loss: {metrics['train_loss'][-1]:.4f} | "
              f"Val Loss: {metrics['val_loss'][-1]:.4f} | "
              f"Train Acc: {metrics['train_acc'][-1]:.4f} | "
              f"Val Acc: {metrics['val_acc'][-1]:.4f}")
    
    return metrics

# Train and evaluate
metrics = train_model(model, criterion, optimizer)

# --------------------------
# 7. Dynamic Visualization
# --------------------------
def plot_metrics(metrics):
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(metrics['train_loss'], label='Train')
    plt.plot(metrics['val_loss'], label='Validation')
    plt.title('Loss Curves')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(metrics['train_acc'], label='Train')
    plt.plot(metrics['val_acc'], label='Validation')
    plt.title('Accuracy Curves')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

plot_metrics(metrics)

# Final evaluation
def evaluate_model(model, loader):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for X_batch, y_batch in loader:
            outputs = model(X_batch)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.numpy())
            all_labels.extend(y_batch.numpy())
    return all_preds, all_labels

# Generate reports
def generate_report(y_true, y_pred, labels):
    print("Classification Report:")
    print(classification_report(y_true, y_pred, target_names=labels))
    
    plt.figure(figsize=(10,8))
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels, yticklabels=labels)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()

# Gradient Boosting Report
print("\nGradient Boosting Results:")
gb_preds = gb_model.predict(X_test_scaled)
generate_report(y_test, gb_preds, label_names)

# DNN Report
print("\nDNN Results:")
dnn_preds, dnn_labels = evaluate_model(model, test_loader)
generate_report(dnn_labels, dnn_preds, label_names)
