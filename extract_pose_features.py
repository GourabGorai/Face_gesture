
import cv2
import mediapipe as mp
import scipy.io
import os
import glob
import csv
import sys
import numpy as np

# SYS PATH FIX
site_pkg = r"D:\BragBoard-main\face_system\Lib\site-packages"
if os.path.exists(site_pkg) and site_pkg not in sys.path:
     sys.path.insert(0, site_pkg)

# Target Mapping
TARGET_CLASSES = {
    'rope skipping': 0, # Jump 
    'ballroom': 1,      # Down
    'boxing': 2,        # Left
    'fencing': 3,       # Right
    'standing': 4       # Idle
}

# Mapping for Flipping
# If we flip 'boxing' (Left), it becomes 'Right' (conceptually, or remains 'Left' but flipped?)
# Logic: 
#   Box left (lean left) -> Flip -> Box right (lean right).
#   So if Label=2 (Left), Flipped Image Label=3 (Right).
#   If Label=3 (Right), Flipped Image Label=2 (Left).
#   Jump (0) -> Flip -> Jump (0)
#   Down (1) -> Flip -> Down (1)
#   Idle (4) -> Flip -> Idle (4)

FLIP_MAP = {
    0: 0,
    1: 1,
    2: 3, # Left becomes Right
    3: 2, # Right becomes Left
    4: 4
}

MAX_SAMPLES = 600

def extract_features():
    base_path = r"d:\BragBoard-main\Face Detection"
    
    sub_dir = os.path.join(base_path, "mpii_human_pose_v1_u12_2")
    mat_files = glob.glob(os.path.join(sub_dir, "*.mat"))
    mat_file = mat_files[0] if mat_files else os.path.join(base_path, "mpii_human_pose_v1_u12_2.mat")
    
    img_dir = os.path.join(base_path, "images")
    output_csv = os.path.join(base_path, "pose_dataset_augmented.csv")

    print(f"Loading {mat_file}...")
    try:
        mat = scipy.io.loadmat(mat_file, struct_as_record=False, squeeze_me=True)
    except Exception as e:
        print(f"Error loading MAT: {e}")
        return

    release = mat['RELEASE']
    annolist = release.annolist
    acts = release.act
    
    mp_pose = mp.solutions.pose
    
    header = ['label']
    for i in range(33):
        header.extend([f'x{i}', f'y{i}', f'z{i}', f'v{i}'])
        
    print(f"Writing augmented features to {output_csv}...")
    
    with open(output_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        
        counts = {k:0 for k in TARGET_CLASSES.keys()}
        
        with mp_pose.Pose(static_image_mode=True, 
                          min_detection_confidence=0.5, 
                          model_complexity=1) as pose:
            
            total = len(annolist)
            print(f"Scanning {total} images...")
            
            for i in range(total):
                if i >= len(acts): break
                try:
                    act_entry = acts[i]
                    if hasattr(act_entry, 'act_name') and isinstance(act_entry.act_name, str):
                        activity = act_entry.act_name.lower()
                        
                        target_id = None
                        matched_key = None
                        
                        for key, tid in TARGET_CLASSES.items():
                            if key in activity:
                                target_id = tid
                                matched_key = key
                                break
                        
                        if target_id is not None:
                            # Check limit (soft limit, since we augment)
                            if counts[matched_key] >= MAX_SAMPLES:
                                continue
                                
                            img_name = annolist[i].image.name
                            img_path = os.path.join(img_dir, img_name)
                            
                            if os.path.exists(img_path):
                                image = cv2.imread(img_path)
                                if image is None: continue
                                
                                # Process Original
                                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                                results = pose.process(image_rgb)
                                
                                if results.pose_landmarks:
                                    row = [target_id]
                                    for landmark in results.pose_landmarks.landmark:
                                        row.extend([landmark.x, landmark.y, landmark.z, landmark.visibility])
                                    writer.writerow(row)
                                    counts[matched_key] += 1
                                
                                # Process Flipped (Augmentation)
                                # Flip horizontally
                                image_flipped = cv2.flip(image, 1)
                                image_flipped_rgb = cv2.cvtColor(image_flipped, cv2.COLOR_BGR2RGB)
                                
                                results_flip = pose.process(image_flipped_rgb)
                                
                                if results_flip.pose_landmarks:
                                    # Determine new label
                                    flipped_id = FLIP_MAP.get(target_id, target_id)
                                    
                                    row_flip = [flipped_id]
                                    for landmark in results_flip.pose_landmarks.landmark:
                                        row_flip.extend([landmark.x, landmark.y, landmark.z, landmark.visibility])
                                    writer.writerow(row_flip)
                                    
                                    # Update count for the *new* class if it changed
                                    # But to keep logic simple, we just log progress
                                    
                                    if sum(counts.values()) % 50 == 0:
                                        print(f"Extracted {sum(counts.values())} original samples...")
                except Exception as e:
                    continue

    print("Augmented Extraction Complete.")
    print("Original Counts:", counts)

if __name__ == "__main__":
    extract_features()
