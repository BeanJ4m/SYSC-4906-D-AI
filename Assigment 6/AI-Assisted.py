import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Load and preprocess data
df = pd.read_csv("Assigment 6/Eight_Class_Dataset.csv").dropna()
target_col = 'Attack_type'
label_names = sorted(df[target_col].unique())
label_mapping = {name: idx for idx, name in enumerate(label_names)}
df[target_col] = df[target_col].map(label_mapping)

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
    return (
        DataLoader(train, batch_size=batch_size, shuffle=True),
        DataLoader(val, batch_size=batch_size, shuffle=False),
        DataLoader(test, batch_size=batch_size, shuffle=False),
    )

# DNN Architectures
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

class CustomDNN(nn.Module):
    def __init__(self, input_size, output_size):
        super(CustomDNN, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, output_size)
        )
    
    def forward(self, x):
        return self.layers(x)

# Initialize models
model = DynamicDNN(df.shape[1] - 1, len(label_names))
custom_model = CustomDNN(df.shape[1] - 1, len(label_names))
criterion = nn.CrossEntropyLoss()

# Training function
def train_model(model, criterion, optimizer, train_loader, val_loader, epochs=20):
    metrics = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    for epoch in range(epochs):
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
        
        model.eval()
        val_loss, val_correct = 0, 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                val_loss += loss.item() * X_batch.size(0)
                _, preds = torch.max(outputs, 1)
                val_correct += (preds == y_batch).sum().item()
        
        metrics['train_loss'].append(train_loss / len(train_loader.dataset))
        metrics['val_loss'].append(val_loss / len(val_loader.dataset))
        metrics['train_acc'].append(train_correct / len(train_loader.dataset))
        metrics['val_acc'].append(val_correct / len(val_loader.dataset))
        
        print(f"Epoch {epoch+1}/{epochs} | "
              f"Train Loss: {metrics['train_loss'][-1]:.4f} | "
              f"Val Loss: {metrics['val_loss'][-1]:.4f} | "
              f"Train Acc: {metrics['train_acc'][-1]:.4f} | "
              f"Val Acc: {metrics['val_acc'][-1]:.4f}")
    return metrics

# Plot metrics
def plot_metrics(metrics, filename_prefix):
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(metrics['train_loss'], label='Train Loss')
    plt.plot(metrics['val_loss'], label='Validation Loss')
    plt.title(f'{filename_prefix}: Loss Curves')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(metrics['train_acc'], label='Train Accuracy')
    plt.plot(metrics['val_acc'], label='Validation Accuracy')
    plt.title(f'{filename_prefix}: Accuracy Curves')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig(f"{filename_prefix}_loss_accuracy_curves.png")
    plt.clf()

# Evaluate model
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
def generate_report(y_true, y_pred, labels, filename_prefix):
# Dynamically filter labels based on the classes present in y_true
    # Dynamically filter labels based on the classes present in y_true
    unique_classes = sorted(np.unique(y_true))
    filtered_labels = [labels[i] for i in unique_classes]

    print("Classification Report:")
    print(classification_report(y_true, y_pred, labels=unique_classes, target_names=filtered_labels))
    
    # Confusion Matrix
    plt.figure(figsize=(10, 8))
    cm = confusion_matrix(y_true, y_pred, labels=unique_classes)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=filtered_labels, yticklabels=filtered_labels)
    plt.title(f'{filename_prefix}: Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    
    # Adjust layout to prevent labels from being cut off
    plt.tight_layout()
    
    # Save the confusion matrix
    plt.savefig(f"{filename_prefix}_confusion_matrix.png")
    plt.clf()

# Logistic Regression with SGD
sgd_model = SGDClassifier(loss='log_loss', max_iter=1, learning_rate='constant', eta0=0.01, random_state=42, warm_start=True)

def train_sgd_model(X_train, y_train, X_val, y_val, epochs=20):
    train_accs, val_accs = [], []
    for epoch in range(epochs):
        sgd_model.fit(X_train, y_train)
        train_acc = sgd_model.score(X_train, y_train)
        val_acc = sgd_model.score(X_val, y_val)
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        print(f"Epoch {epoch+1}/{epochs} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")
    return train_accs, val_accs

# Run experiments
def run_experiment(filtered_df, filename_prefix, learning_rates=None, batch_sizes=None):
    X = filtered_df.drop(target_col, axis=1)
    y = filtered_df[target_col]
    X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.25, random_state=42, stratify=y_train_val)
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    train_dataset = DynamicDataset(X_train_scaled, y_train.values)
    val_dataset = DynamicDataset(X_val_scaled, y_val.values)
    test_dataset = DynamicDataset(X_test_scaled, y_test.values)

    # Gradient Boosting
    print("\n--- Gradient Boosting ---")
    gb_model = GradientBoostingClassifier(n_estimators=20, learning_rate=0.1, random_state=42)
    gb_model.fit(X_train_scaled, y_train)
    gb_preds = gb_model.predict(X_test_scaled)
    # Generate confusion matrix and classification report for Gradient Boosting
    generate_report(y_test, gb_preds, label_names, f"{filename_prefix}_gb")

    # Default settings for DNN
    print("\n--- DNN (Default Settings) ---")
    train_loader, val_loader, test_loader = create_loaders(train_dataset, val_dataset, test_dataset)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    metrics = train_model(model, criterion, optimizer, train_loader, val_loader)
    plot_metrics(metrics, f"{filename_prefix}_dnn")
    dnn_preds, dnn_labels = evaluate_model(model, test_loader)
    # Generate confusion matrix and classification report for DNN (default settings)
    generate_report(dnn_labels, dnn_preds, label_names, f"{filename_prefix}_dnn")

    # Vary learning rates for DNN
    if learning_rates:
        for lr in learning_rates:
            print(f"\nTraining with learning rate: {lr}")
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
            metrics = train_model(model, criterion, optimizer, train_loader, val_loader)
            plot_metrics(metrics, f"{filename_prefix}_lr_{lr}")
            dnn_preds, dnn_labels = evaluate_model(model, test_loader)
            # Generate confusion matrix and classification report for DNN (learning rate variation)
            generate_report(dnn_labels, dnn_preds, label_names, f"{filename_prefix}_lr_{lr}")

    # Vary batch sizes for DNN
    if batch_sizes:
        for batch_size in batch_sizes:
            print(f"\nTraining with batch size: {batch_size}")
            train_loader, val_loader, test_loader = create_loaders(train_dataset, val_dataset, test_dataset, batch_size=batch_size)
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
            metrics = train_model(model, criterion, optimizer, train_loader, val_loader)
            plot_metrics(metrics, f"{filename_prefix}_bs_{batch_size}")
            dnn_preds, dnn_labels = evaluate_model(model, test_loader)
            # Generate confusion matrix and classification report for DNN (batch size variation)
            generate_report(dnn_labels, dnn_preds, label_names, f"{filename_prefix}_bs_{batch_size}")

# Scenarios
def scenario_1():
    print("\n--- Scenario 1: Full Dataset Training and Evaluation ---")
    run_experiment(df, "scenario_1")

def scenario_2():
    print("\n--- Scenario 2: Dataset Reduction and Parameter Variation ---")
    run_experiment(df, "scenario_2_full", learning_rates=[0.01, 0.05, 0.001], batch_sizes=[32, 64])
    reduced_df = df.sample(n=400, random_state=42).reset_index(drop=True)
    run_experiment(reduced_df, "scenario_2_reduced", learning_rates=[0.01, 0.05, 0.001], batch_sizes=[32, 64])

def scenario_3():
    print("\n--- Scenario 3: Class Exclusion Experiment ---")
    class_counts = df[target_col].value_counts()
    top_classes = class_counts.nlargest(4).index
    df_exclude_top = df[~df[target_col].isin(top_classes)]
    run_experiment(df_exclude_top, "scenario_3_exclude_top")
    bottom_classes = class_counts.nsmallest(4).index
    df_exclude_bottom = df[~df[target_col].isin(bottom_classes)]
    run_experiment(df_exclude_bottom, "scenario_3_exclude_bottom")

def scenario_4():
    print("\n--- Scenario 4: Model Architecture & Alternative ML Technique ---")

    # Split and scale data
    X = df.drop(target_col, axis=1)
    y = df[target_col]
    X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.25, random_state=42, stratify=y_train_val)
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    train_dataset = DynamicDataset(X_train_scaled, y_train.values)
    val_dataset = DynamicDataset(X_val_scaled, y_val.values)
    test_dataset = DynamicDataset(X_test_scaled, y_test.values)
    train_loader, val_loader, test_loader = create_loaders(train_dataset, val_dataset, test_dataset)

    # Original DNN
    print("\n--- Original DNN ---")
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    metrics = train_model(model, criterion, optimizer, train_loader, val_loader)
    plot_metrics(metrics, "scenario_4_original_dnn")
    dnn_preds, dnn_labels = evaluate_model(model, test_loader)
    generate_report(dnn_labels, dnn_preds, label_names, "scenario_4_original_dnn")

    # Custom DNN
    print("\n--- Custom DNN ---")
    custom_optimizer = torch.optim.Adam(custom_model.parameters(), lr=0.001)
    custom_metrics = train_model(custom_model, criterion, custom_optimizer, train_loader, val_loader)
    plot_metrics(custom_metrics, "scenario_4_custom_dnn")
    custom_dnn_preds, custom_dnn_labels = evaluate_model(custom_model, test_loader)
    generate_report(custom_dnn_labels, custom_dnn_preds, label_names, "scenario_4_custom_dnn")

    # Logistic Regression with SGD
    print("\n--- Logistic Regression with SGD ---")
    sgd_train_accs, sgd_val_accs = train_sgd_model(X_train_scaled, y_train, X_val_scaled, y_val, epochs=20)
    plt.figure(figsize=(8, 6))
    plt.plot(sgd_train_accs, label='Train Accuracy')
    plt.plot(sgd_val_accs, label='Validation Accuracy')
    plt.title("Logistic Regression with SGD: Accuracy Curves")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.savefig("scenario_4_sgd_accuracy_curves.png")
    plt.clf()
    sgd_preds = sgd_model.predict(X_test_scaled)
    generate_report(y_test, sgd_preds, label_names, "scenario_4_sgd")

    # Plot combined accuracy curves
    plot_combined_accuracy_curves(metrics, custom_metrics, sgd_train_accs, sgd_val_accs)

    # Generate confusion matrices
    original_cm = confusion_matrix(y_test, dnn_preds)
    custom_cm = confusion_matrix(y_test, custom_dnn_preds)
    sgd_cm = confusion_matrix(y_test, sgd_preds)

    # Plot combined confusion matrices
    plot_combined_confusion_matrices(original_cm, custom_cm, sgd_cm, label_names)

    generate_classification_report_table(y_test, dnn_preds, custom_dnn_preds, sgd_preds, label_names)

# Plot combined accuracy curves
def plot_combined_accuracy_curves(original_metrics, custom_metrics, sgd_train_accs, sgd_val_accs):
    plt.figure(figsize=(10, 6))

    # Original DNN
    plt.plot(original_metrics['train_acc'], label='Original DNN - Train', linestyle='--')
    plt.plot(original_metrics['val_acc'], label='Original DNN - Validation')

    # Custom DNN
    plt.plot(custom_metrics['train_acc'], label='Custom DNN - Train', linestyle='--')
    plt.plot(custom_metrics['val_acc'], label='Custom DNN - Validation')

    # Logistic Regression with SGD
    plt.plot(sgd_train_accs, label='SGD - Train', linestyle='--')
    plt.plot(sgd_val_accs, label='SGD  - Validation')

    # Add labels, legend, and title
    plt.title("Accuracy Curves Comparison")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True)
    plt.savefig("combined_accuracy_curves.png")

# Plot combined confusion matrices
def plot_combined_confusion_matrices(original_cm, custom_cm, sgd_cm, labels):
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Original DNN Confusion Matrix
    sns.heatmap(original_cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels, ax=axes[0])
    axes[0].set_title("Original DNN")
    axes[0].set_xlabel("Predicted")
    axes[0].set_ylabel("True")

    # Custom DNN Confusion Matrix
    sns.heatmap(custom_cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels, ax=axes[1])
    axes[1].set_title("Custom DNN")
    axes[1].set_xlabel("Predicted")
    axes[1].set_ylabel("True")

    # Logistic Regression with SGD Confusion Matrix
    sns.heatmap(sgd_cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels, ax=axes[2])
    axes[2].set_title("Logistic Regression (SGD)")
    axes[2].set_xlabel("Predicted")
    axes[2].set_ylabel("True")

    # Adjust layout and save
    plt.tight_layout()
    plt.savefig("combined_confusion_matrices.png")

from sklearn.metrics import classification_report
import pandas as pd

def generate_classification_report_table(y_test, dnn_preds, custom_dnn_preds, sgd_preds, labels):
    # Generate classification reports
    original_report = classification_report(y_test, dnn_preds, target_names=labels, output_dict=True)
    custom_report = classification_report(y_test, custom_dnn_preds, target_names=labels, output_dict=True)
    sgd_report = classification_report(y_test, sgd_preds, target_names=labels, output_dict=True)

    # Extract macro avg and weighted avg
    data = {
        "Model": ["Original DNN", "Custom DNN", "Logistic Regression (SGD)"],
        "Precision (Macro Avg)": [
            original_report["macro avg"]["precision"],
            custom_report["macro avg"]["precision"],
            sgd_report["macro avg"]["precision"]
        ],
        "Recall (Macro Avg)": [
            original_report["macro avg"]["recall"],
            custom_report["macro avg"]["recall"],
            sgd_report["macro avg"]["recall"]
        ],
        "F1-Score (Macro Avg)": [
            original_report["macro avg"]["f1-score"],
            custom_report["macro avg"]["f1-score"],
            sgd_report["macro avg"]["f1-score"]
        ],
        "Accuracy": [
            original_report["accuracy"],
            custom_report["accuracy"],
            sgd_report["accuracy"]
        ]
    }

    # Create a DataFrame
    df = pd.DataFrame(data)
    print(df)

    # Save to CSV
    df.to_csv("classification_report_comparison.csv", index=False)

# Main Execution
if __name__ == "__main__":
    scenario_1()
    scenario_2()
    scenario_3()
    scenario_4()

