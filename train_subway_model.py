
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, Dataset
from torchvision import transforms, models
from PIL import Image
import scipy.io
import pickle

# Selected Classes for Game Control
TARGET_CLASSES = {
    'rope skipping': 0, # Up
    'ballroom': 1,      # Down
    'boxing': 2,        # Left
    'fencing': 3        # Right
}

class MPIISubwayDataset(Dataset):
    def __init__(self, root_dir, mat_file, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.mat_file = mat_file
        self.data = []
        self._load_annotations()

    def _load_annotations(self):
        print(f"Loading annotations from {self.mat_file}...")
        try:
            mat = scipy.io.loadmat(self.mat_file, struct_as_record=False, squeeze_me=True)
            release = mat['RELEASE']
            annolist = release.annolist
            acts = release.act
            
            for i in range(len(annolist)):
                if i >= len(acts): break
                try:
                    act_entry = acts[i]
                    if hasattr(act_entry, 'act_name') and isinstance(act_entry.act_name, str):
                        activity = act_entry.act_name
                        
                        # Filter for target classes
                        target_id = None
                        for key, tid in TARGET_CLASSES.items():
                            if key in activity:
                                target_id = tid
                                break
                        
                        if target_id is not None:
                            img_name = annolist[i].image.name
                            self.data.append({
                                'image': img_name,
                                'label': target_id,
                                'original_activity': activity
                            })
                except:
                    continue
            
            print(f"Filtered Dataset: {len(self.data)} samples.")
            for k, v in TARGET_CLASSES.items():
                count = sum(1 for x in self.data if x['label'] == v)
                print(f"  {k}: {count}")
                
        except Exception as e:
            print(f"Error: {e}")
            raise e

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        img_name = item['image']
        label = item['label']
        
        img_path = os.path.join(self.root_dir, "images", img_name)
        
        try:
            image = Image.open(img_path).convert('RGB')
        except:
            image = Image.new('RGB', (224, 224))
            
        if self.transform:
            image = self.transform(image)
            
        return image, label

def train_subway_model():
    if not torch.cuda.is_available():
        print("CUDA required.")
        return
        
    device = torch.device('cuda')
    batch_size = 16
    lr = 0.001
    epochs = 10
    
    base_path = r"d:\BragBoard-main\Face Detection"
    # Dynamic finding of mat file
    import glob
    sub_dir = os.path.join(base_path, "mpii_human_pose_v1_u12_2")
    mat_files = glob.glob(os.path.join(sub_dir, "*.mat"))
    mat_file = mat_files[0] if mat_files else os.path.join(base_path, "mpii_human_pose_v1_u12_2.mat")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    dataset = MPIISubwayDataset(base_path, mat_file, transform)
    
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    num_classes = 4
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    print("Starting training...")
    best_acc = 0.0
    
    for epoch in range(epochs):
        model.train()
        for i, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        # Validation
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        acc = 100 * correct / total
        print(f"Epoch {epoch+1}/{epochs}, Val Acc: {acc:.2f}%")
        
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), 'subway_model.pth')
            print("Saved best subway_model.pth")

if __name__ == "__main__":
    train_subway_model()
