
import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms, models
from mpii_utils import MPIIDataset

def train_model(args):
    # Device configuration
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available! User requested forced GPU training.")
    device = torch.device('cuda')
    print(f"Using device: {device}")
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")

    # Hyperparameters
    num_epochs = args.epochs
    batch_size = args.batch_size
    learning_rate = args.learning_rate

    # Data transformations
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Load Dataset
    # Dynamic finding of mat file to avoid path issues
    import glob
    sub_dir = os.path.join(args.data_path, "mpii_human_pose_v1_u12_2")
    mat_files = glob.glob(os.path.join(sub_dir, "*.mat"))
    if mat_files:
        mat_file = mat_files[0]
        print(f"Found annotation file: {mat_file}")
    else:
        # Fallback or error
        mat_file = os.path.join(args.data_path, "mpii_human_pose_v1_u12_2.mat")
        print(f"Warning: Could not find mat file in {sub_dir}. Trying: {mat_file}")

    dataset = MPIIDataset(root_dir=args.data_path, mat_file=mat_file, transform=transform)

    # Split dataset
    # Use a smaller subset if testing (dry-run)
    if args.dry_run:
        print("Dry run mode: using small subset")
        subset_size = 100
        remain_size = len(dataset) - subset_size
        dataset, _ = random_split(dataset, [subset_size, remain_size])

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # Optimize data loading for GPU
    num_workers = 4 if os.name != 'nt' else 0 # Windows often has issues with multiprocessing in interactive shells, sticking to 0 for safety or 2 if user insists.
    # Actually, let's try 2 workers and pin_memory for speed, keeping in mind Windows limitations
    # If standard windows fork issues arise, set workers to 0.
    # User asked for FORCE, so they want speed.
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=0)
    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=0)

    num_classes = len(dataset.dataset.activity_map) if hasattr(dataset, 'dataset') else len(dataset.activity_map)
    print(f"Number of classes: {num_classes}")

    # Optimize for fixed input size
    torch.backends.cudnn.benchmark = True # Good for fixed size inputs like (224,224)

    # Save Class Mapping
    import pickle
    map_path = 'activity_map.pkl'
    # Handle subset nesting if needed, but 'dataset' here is the root MPIIDataset (or we use the ref before split)
    # The 'dataset' variable might be overwritten by random_split in dry-run, so we should be careful.
    # Actually, in dry_run block: dataset, _ = random_split(dataset...)
    # So we should save it before dry run split or access deeply.
    # Let's clean this up: The original dataset object has the map.
    
    # We'll save it right here, assuming 'dataset' is still accessible or we kept a ref.
    # But wait, line 48 overwrites 'dataset' in dry_run.
    # We should grab the map from the root dataset instance.
    if isinstance(dataset, torch.utils.data.Subset):
         root_ds = dataset.dataset
         while isinstance(root_ds, torch.utils.data.Subset):
             root_ds = root_ds.dataset
         activity_map = root_ds.activity_map
    else:
         activity_map = dataset.activity_map

    with open(map_path, 'wb') as f:
        pickle.dump(activity_map, f)
    print(f"Saved activity map to {map_path}")

    # Model
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    
    # Modify the last layer
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    model = model.to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Train
    best_acc = 0.0
    total_step = len(train_loader)
    for epoch in range(num_epochs):
        model.train()
        for i, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i+1) % 10 == 0:
                print (f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{total_step}], Loss: {loss.item():.4f}')

        # Validation
        model.eval()
        with torch.no_grad():
            correct = 0
            total = 0
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            acc = 100 * correct / total
            print(f'Accuracy of the model on validation images: {acc} %')
            
            if acc > best_acc:
                best_acc = acc
                torch.save(model.state_dict(), 'best_activity_model.pth')
                print(f"New best model saved with accuracy: {acc}%")

    # Save the final model
    torch.save(model.state_dict(), 'activity_model.pth')
    print("Final model saved to activity_model.pth")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train Activity Recognition Model')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--learning-rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--data-path', type=str, default='d:/BragBoard-main/Face Detection', help='Path to dataset')
    parser.add_argument('--dry-run', action='store_true', help='Run with small data for verification')
    
    args = parser.parse_args()
    train_model(args)
