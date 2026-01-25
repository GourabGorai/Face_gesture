
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
import pickle
import os
import sys

# Train a simple MLP on the extracted MediaPipe landmarks

class PoseDataset(Dataset):
    def __init__(self, csv_file):
        try:
            self.data = pd.read_csv(csv_file)
            # Labels: Ensure int64 for PyTorch LongTensor
            self.y = self.data.iloc[:, 0].values.astype('int64')
            # Features: float32
            self.x = self.data.iloc[:, 1:].values.astype('float32') 
        except Exception as e:
            print(f"Error loading CSV: {e}")
            self.x = []
            self.y = []

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        # Return tensors explicitly
        return torch.tensor(self.x[idx], dtype=torch.float32), torch.tensor(self.y[idx], dtype=torch.long)

class PoseClassifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(PoseClassifier, self).__init__()
        # Input: 33 landmarks * 4 values (x,y,z,vis) = 132
        self.fc1 = nn.Linear(input_dim, 128)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.2)
        
        self.fc2 = nn.Linear(128, 64)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(0.2)
        
        self.fc3 = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.dropout2(x)
        x = self.fc3(x)
        return x

def train_subway_model():
    base_path = r"d:\BragBoard-main\Face Detection"
    
    # Priority: Check for custom user data first
    custom_csv = os.path.join(base_path, "my_pose_data.csv")
    aug_csv = os.path.join(base_path, "pose_dataset_augmented.csv")
    
    if os.path.exists(custom_csv):
        print(f"Found Custom User Data: {custom_csv}")
        csv_file = custom_csv
    elif os.path.exists(aug_csv):
        print(f"Using Augmented Dataset: {aug_csv}")
        csv_file = aug_csv
    else:
        print("No dataset found! Run capture_data.py first.")
        return
    
    model_path = os.path.join(base_path, "subway_pose_model.pth")
    map_path = os.path.join(base_path, "pose_label_map.pkl")
    
    if not os.path.exists(csv_file):
        print(f"Dataset not found at {csv_file}")
        return

    # MAPPING
    label_map = {
        0: 'JUMP',
        1: 'DOWN',
        2: 'LEFT',
        3: 'RIGHT',
        4: 'IDLE'
    }
    
    with open(map_path, 'wb') as f:
        pickle.dump(label_map, f)
    print(f"Saved label map to {map_path}")

    # Data
    dataset = PoseDataset(csv_file)
    if len(dataset) == 0:
        print("Empty dataset!")
        return

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    # Model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on {device}")
    
    model = PoseClassifier(input_dim=132, num_classes=5)
    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    epochs = 50
    best_acc = 0.0
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for features, labels in train_loader:
            features, labels = features.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
        # Validation
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for features, labels in val_loader:
                features, labels = features.to(device), labels.to(device)
                outputs = model(features)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        acc = 100 * correct / total
        
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), model_path)
        
        if (epoch+1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs} | Loss: {train_loss/len(train_loader):.4f} | Val Acc: {acc:.2f}%")

    print(f"Training Complete. Best Accuracy: {best_acc:.2f}%")
    print(f"Model saved to {model_path}")

if __name__ == "__main__":
    train_subway_model()
