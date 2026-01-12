import os
import shutil
import cv2
import json
import yaml
import glob
import torch
from tqdm import tqdm
from ultralytics import YOLO

# --- CONFIGURATION ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ARCHIVE_DIR = os.path.join(BASE_DIR, 'archive')

# WIDER FACE PATHS
WIDER_TRAIN_IMG = os.path.join(BASE_DIR, 'Face Dataset', 'WIDER_train', 'WIDER_train', 'images')
WIDER_VAL_IMG = os.path.join(BASE_DIR, 'Face Dataset', 'WIDER_val', 'WIDER_val', 'images')
WIDER_TRAIN_LABELS = os.path.join(BASE_DIR, 'Face Dataset', 'WIDER_train', 'WIDER_train', 'labels')
WIDER_VAL_LABELS = os.path.join(BASE_DIR, 'Face Dataset', 'WIDER_val', 'WIDER_val', 'labels')

# HaGRID PATHS
HAGRID_ROOT = os.path.join(BASE_DIR, 'Hand Gesture')
HAGRID_IMG_DIR = HAGRID_ROOT # Images extracted here
HAGRID_ANN_DIR = os.path.join(HAGRID_ROOT, 'ann_train_val')
HAGRID_LABELS = os.path.join(HAGRID_ROOT, 'labels')

# OUTPUTS
YAML_FILE = os.path.join(BASE_DIR, 'custom_config.yaml')
FINAL_MODEL_NAME = "final_face_gesture_model.pt"

# CLASS MAPPING
CLASS_MAP = {
    "face": 0,
    "call": 1, "dislike": 2, "fist": 3, "four": 4, "like": 5,
    "mute": 6, "ok": 7, "one": 8, "palm": 9, "peace": 10,
    "peace_inverted": 11, "rock": 12, "stop": 13, "three": 14,
    "three2": 15, "two_up": 16, "two_up_inverted": 17, "no_gesture": 18
}

def check_gpu_status():
    print("\n--- SYSTEM CHECK ---")
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        print(f"✅ SUCCESS: GPU Detected: {gpu_name}")
    else:
        print("⚠️ WARNING: No GPU detected. Training will be SLOW.")
    print("--------------------\n")

def convert_bbox(x, y, w, h, img_w=1, img_h=1, mode='corner'):
    if mode == 'corner': # WIDER
        x_center = (x + w / 2) / img_w
        y_center = (y + h / 2) / img_h
        norm_w = w / img_w
        norm_h = h / img_h
    elif mode == 'hagrid': # HaGRID (already normalized 0-1 relative)
        x_center = x + (w / 2)
        y_center = y + (h / 2)
        norm_w = w
        norm_h = h

    return (max(0, min(1, x_center)), max(0, min(1, y_center)), 
            max(0, min(1, norm_w)), max(0, min(1, norm_h)))

def process_wider(subset):
    img_root = WIDER_TRAIN_IMG if subset == 'train' else WIDER_VAL_IMG
    out_dir = WIDER_TRAIN_LABELS if subset == 'train' else WIDER_VAL_LABELS
    
    # Locate Annotation File
    # Original paths from user structure
    ann_file_name = 'wider_face_train_bbx_gt.txt' if subset == 'train' else 'wider_face_val_bbx_gt.txt'
    # Check common locations
    candidates = [
        os.path.join(BASE_DIR, 'Face Dataset', 'wider_face_annotations', 'wider_face_split', ann_file_name),
        os.path.join(ARCHIVE_DIR, 'wider_face_annotations', 'wider_face_split', ann_file_name)
    ]
    gt_file = next((f for f in candidates if os.path.exists(f)), None)
    
    if not gt_file:
        print(f"⚠️ Could not find annotations for {subset}. Skipping.")
        return

    print(f"Processing WIDER {subset}...")
    os.makedirs(out_dir, exist_ok=True)
    
    # Check if already done (simple check)
    if len(glob.glob(os.path.join(out_dir, "**", "*.txt"), recursive=True)) > 100:
        print(f"✅ WIDER {subset} labels seem to exist. Skipping conversion.")
        return

    with open(gt_file, 'r') as f:
        lines = f.readlines()

    i = 0
    count = 0
    pbar = tqdm(total=len(lines), desc=f"WIDER {subset}")
    
    while i < len(lines):
        file_name = lines[i].strip()
        i += 1
        pbar.update(1)
        
        if not file_name.endswith('.jpg'): continue
        
        try:
            num_boxes = int(lines[i].strip())
            i += 1
            pbar.update(1)
        except ValueError: continue

        img_path = os.path.join(img_root, file_name)
        if not os.path.exists(img_path):
            i += num_boxes
            pbar.update(num_boxes)
            continue
            
        # Get dimensions
        img = cv2.imread(img_path)
        if img is None:
            i += num_boxes
            pbar.update(num_boxes)
            continue
        h_img, w_img = img.shape[:2]

        label_subdir = os.path.dirname(file_name)
        label_name = os.path.splitext(os.path.basename(file_name))[0] + ".txt"
        save_dir = os.path.join(out_dir, label_subdir)
        os.makedirs(save_dir, exist_ok=True)
        
        yolo_data = []
        for _ in range(num_boxes):
            box = lines[i].strip().split()
            i += 1
            pbar.update(1)
            x1, y1, w, h = map(int, box[:4])
            xc, yc, nw, nh = convert_bbox(x1, y1, w, h, w_img, h_img, mode='corner')
            yolo_data.append(f"0 {xc:.6f} {yc:.6f} {nw:.6f} {nh:.6f}")
            
        with open(os.path.join(save_dir, label_name), 'w') as f:
            f.write('\n'.join(yolo_data))
        count += 1
        
    pbar.close()
    print(f"Converted {count} WIDER {subset} images.")

def process_hagrid():
    print("\nProcessing HaGRID Dataset...")
    if not os.path.exists(HAGRID_ANN_DIR):
        print("⚠️ HaGRID annotations not found. Skipping.")
        return

    os.makedirs(HAGRID_LABELS, exist_ok=True)
    
    # Check if labels exist
    if len(os.listdir(HAGRID_LABELS)) > 100:
        print("✅ HaGRID labels seem to exist. Skipping conversion.")
        return

    json_files = [f for f in os.listdir(HAGRID_ANN_DIR) if f.endswith('.json')]
    
    for json_file in json_files:
        with open(os.path.join(HAGRID_ANN_DIR, json_file), 'r') as f:
            data = json.load(f)
            
        for img_id, content in tqdm(data.items(), desc=json_file):
            bboxes = content.get('bboxes', [])
            labels = content.get('labels', [])
            
            # Check if image exists (optional optimization: skip check if confident)
            # HaGRID structure after unzip: Hand Gesture/call/image_id.jpg OR Hand Gesture/image_id.jpg
            # Let's assume flattened or we need to find it. 
            # The download script usually creates folders per class? 
            # Subsample usually has structure. We will assume flat for now or update finding logic.
            # Actually, HaGRID subsamples usually put images in class folders.
            
            yolo_data = []
            valid_image = False
            
            # Try to locate image to confirm existence
            # Strategy: We generate label. Training will fail if image is missing, 
            # but we assume unzip is/will be successful.
            
            for j in range(len(bboxes)):
                label = labels[j]
                if label in CLASS_MAP:
                    cls_id = CLASS_MAP[label]
                    x, y, w, h = bboxes[j]
                    xc, yc, nw, nh = convert_bbox(x, y, w, h, mode='hagrid')
                    yolo_data.append(f"{cls_id} {xc:.6f} {yc:.6f} {nw:.6f} {nh:.6f}")
            
            if yolo_data:
                msg = '\n'.join(yolo_data)
                # Save label as {img_id}.txt
                with open(os.path.join(HAGRID_LABELS, f"{img_id}.txt"), 'w') as f:
                    f.write(msg)

def create_config():
    print("\nGenerating YAML config...")
    
    # We list the roots. YOLO allows list of directories.
    # WIDER structure: .../images/0--Parade/...
    # HaGRID structure: .../Hand Gesture (containing class folders or jpgs)
    # Important: YOLO Auto-discovery expects 'images' and 'labels' folders or explicit files.
    # Our WIDER setup: 
    #   images: Face Dataset/WIDER_train/WIDER_train/images
    #   labels: Face Dataset/WIDER_train/WIDER_train/labels
    # This works automatically because 'labels' is sibling of 'images' (mostly) or we need explicit paths.
    
    # For HaGRID, we put labels in 'Hand Gesture/labels'.
    # Images in 'Hand Gesture'. 
    # If HaGRID images are in subfolders (e.g. Hand Gesture/call/xyz.jpg), 
    # and labels in Hand Gesture/labels/xyz.txt, strict YOLO might not match unless we flatten.
    # BUT, let's include the specific paths.
    
    config = {
        'path': BASE_DIR,
        'train': [
            WIDER_TRAIN_IMG,
            HAGRID_IMG_DIR
        ],
        'val': [
            WIDER_VAL_IMG,
            HAGRID_IMG_DIR # Using same for val for now or subset if we had one
        ],
        'names': {v: k for k, v in CLASS_MAP.items()}
    }
    
    with open(YAML_FILE, 'w') as f:
        yaml.dump(config, f, sort_keys=False)
    print(f"Config saved to {YAML_FILE}")

def main():
    check_gpu_status()
    
    # 1. Prepare Data
    process_wider('train')
    process_wider('val')
    process_hagrid()
    
    create_config()
    
    # 2. Train
    print("\n--- STARTING TRAINING ---")
    print("Loading yolo11n.pt...")
    try:
        model = YOLO('yolo11n.pt')
    except:
        print("yolo11n.pt not found, trying yolov8n.pt...")
        model = YOLO('yolov8n.pt')
        
    print(f"Training on device=0 (GPU) for 10 epochs. Saving to {FINAL_MODEL_NAME}...")
    
    model.train(
        data=YAML_FILE,
        epochs=10, 
        imgsz=640,
        device=0,
        batch=8,  # Increased to 32 to utilize ~8GB VRAM
        project='runs',
        name='face_gesture_train',
        exist_ok=True
    )
    
    # 3. Save Final
    print("\n--- SAVING MODEL ---")
    best = os.path.join('runs', 'face_gesture_train', 'weights', 'best.pt')
    if os.path.exists(best):
        shutil.copy(best, FINAL_MODEL_NAME)
        print(f"SUCCESS: Model saved to {FINAL_MODEL_NAME}")
    else:
        print("Warning: best.pt not found.")

if __name__ == "__main__":
    main()