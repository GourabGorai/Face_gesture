
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
import pickle
import os

class PoseDataset(Dataset):
    def __init__(self, csv_file):
        self.data = pd.read_csv(csv_file)
        self.x = self.data.iloc[:, 1:].values.astype('float32') # Features
        self.y = self.data.iloc[:, 0].values.astype('long')     # Labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

class PoseClassifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(PoseClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.2)
        
        self.fc2 = nn.Linear(128, 64)
        self.relu2 = nn.ReLU()
        
        self.fc3 = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        return x

def train_model():
    base_path = r"d:\BragBoard-main\Face Detection"
    csv_file = os.path.join(base_path, "pose_dataset.csv")
    model_path = os.path.join(base_path, "pose_classifier.pth")
    map_path = os.path.join(base_path, "pose_label_map.pkl")
    
    if not os.path.exists(csv_file):
        print("Dataset not found!")
        return

    label_map = {
        0: 'JUMP',
        1: 'DOWN',
        2: 'LEFT',
        3: 'RIGHT',
        4: 'IDLE'
    }
    
    with open(map_path, 'wb') as f:
        pickle.dump(label_map, f)

    dataset = PoseDataset(csv_file)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    model = PoseClassifier(input_dim=132, num_classes=5)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    print("Starting Training...")
    epochs = 50
    best_acc = 0.0
    
    for epoch in range(epochs):
        model.train()
        for features, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for features, labels in val_loader:
                outputs = model(features)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        acc = 100 * correct / total
        
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), model_path)
            
        if (epoch+1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Val Acc: {acc:.2f}%")
            
    print(f"Best Validation Accuracy: {best_acc:.2f}%")
    print(f"Model saved to {model_path}")

if __name__ == "__main__":
    train_model()
