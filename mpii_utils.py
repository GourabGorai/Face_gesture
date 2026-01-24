
import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import scipy.io

class MPIIDataset(Dataset):
    def __init__(self, root_dir, mat_file, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.mat_file = mat_file
        self.data = []
        self.activity_map = {}
        self._load_annotations()

    def _load_annotations(self):

        print(f"Loading annotations from {self.mat_file}...")
        try:
            mat = scipy.io.loadmat(self.mat_file, struct_as_record=False, squeeze_me=True)
            release = mat['RELEASE']
            annolist = release.annolist
            
            # This is a simplified traversal; the actual structure is complex
            # We iterate to find entries with activity labels
            
            # Note: The structure of MPII is usually RELEASE.annolist(i).image.name
            # and RELEASE.act(i).act_name / act_id
            
            # Need to align with actual structure. 
            # Assuming 'act' field exists in release structure or aligned with annolist
            
            if hasattr(release, 'act'):
                acts = release.act
                for i in range(len(annolist)):
                    try:
                        # check if activity is present
                        # Use resilient access
                        if i < len(acts):
                            act_entry = acts[i]
                            # Check if act_name or cat_name exists and is not empty
                            if hasattr(act_entry, 'act_name') and isinstance(act_entry.act_name, str) and act_entry.act_name:
                                img_name = annolist[i].image.name
                                activity = act_entry.act_name
                                cat_name = act_entry.cat_name if hasattr(act_entry, 'cat_name') else "Unknown"
                                
                                # Store data
                                self.data.append({
                                    'image': img_name,
                                    'activity': activity,
                                    'category': cat_name
                                })
                                
                                # build map
                                if activity not in self.activity_map:
                                    self.activity_map[activity] = len(self.activity_map)
                    except Exception as e:
                        continue # Skip malformed entries
            
            print(f"Loaded {len(self.data)} samples with activity labels.")
            print(f"Found {len(self.activity_map)} unique activities.")
            
        except Exception as e:
            print(f"Error loading .mat file: {e}")
            raise e

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        img_name = item['image']
        activity = item['activity']
        label_id = self.activity_map[activity]
        
        img_path = os.path.join(self.root_dir, "images", img_name)
        
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception:
            # Return a dummy image or handle error if file missing (common in datasets)
            # For robustness, we might want to skip, but Dataset expects return
            # We'll just return a black image or let it fail if critical
            print(f"Warning: Could not load {img_path}")
            image = Image.new('RGB', (224, 224))
            
        if self.transform:
            image = self.transform(image)
            
        return image, label_id
