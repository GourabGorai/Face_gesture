import os
import json
from tqdm import tqdm

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
HAGRID_ROOT = os.path.join(BASE_DIR, 'Hand Gesture')
HAGRID_IMG_ROOT = os.path.join(HAGRID_ROOT, 'yolo_format', 'images')
HAGRID_LABEL_ROOT = os.path.join(HAGRID_ROOT, 'yolo_format', 'labels')

# Prioritize full annotations for complete dataset
HAGRID_ANN_DIR = os.path.join(HAGRID_ROOT, 'ann_train_val')
HAGRID_TEST_ANN_DIR = os.path.join(HAGRID_ROOT, 'ann_test')

if not os.path.exists(HAGRID_ANN_DIR):
    # Fallback if full not present (though user implies it is)
    HAGRID_ANN_DIR = os.path.join(HAGRID_ROOT, 'ann_subsample')


# Class Map (Correct one from test.py)
CLASS_MAP = {
    "face": 0,
    "call": 1, "dislike": 2, "fist": 3, "four": 4, "like": 5,
    "mute": 6, "ok": 7, "one": 8, "palm": 9, "peace": 10,
    "peace_inverted": 11, "rock": 12, "stop": 13, "three": 14,
    "three2": 15, "two_up": 16, "two_up_inverted": 17, "no_gesture": 18,
    "stop_inverted": 19
}

def convert_bbox(x, y, w, h):
    """
    Convert HaGRID bbox (top_left_x, top_left_y, width, height) 
    to YOLO bbox (center_x, center_y, width, height).
    All values are normalized (0-1).
    """
    x_center = x + (w / 2)
    y_center = y + (h / 2)
    return (max(0, min(1, x_center)), max(0, min(1, y_center)), 
            max(0, min(1, w)), max(0, min(1, h)))

def main():
    ann_dirs = [HAGRID_ANN_DIR]
    if os.path.exists(HAGRID_TEST_ANN_DIR):
        ann_dirs.append(HAGRID_TEST_ANN_DIR)
        
    print(f"Using annotation directories: {ann_dirs}")
    
    # Collect all JSONs
    all_jsons = []
    for d in ann_dirs:
        if os.path.exists(d):
            for f in os.listdir(d):
                if f.endswith('.json'):
                    all_jsons.append(os.path.join(d, f))

    splits = ['train', 'val', 'test']
    
    count = 0
    written_files = set()
    unknown_labels = set()
    
    print(f"Scanning images in: {HAGRID_IMG_ROOT}")
    
    for json_path in all_jsons:
        with open(json_path, 'r') as f:
            data = json.load(f)
            
        print(f"Processing {os.path.basename(json_path)}...")
        for img_id, content in tqdm(data.items()):
            bboxes = content.get('bboxes', [])
            labels = content.get('labels', [])
            
            # Find where the image is located
            found_split = None
            for s in splits:
                img_path = os.path.join(HAGRID_IMG_ROOT, s, f"{img_id}.jpg")
                if os.path.exists(img_path):
                    found_split = s
                    break
            
            if not found_split:
                continue

            # Prepare Label File Path
            label_dir = os.path.join(HAGRID_LABEL_ROOT, found_split)
            os.makedirs(label_dir, exist_ok=True)
            label_file = os.path.join(label_dir, f"{img_id}.txt")
            
            lines = []
            for i, label in enumerate(labels):
                if label in CLASS_MAP:
                    cls_id = CLASS_MAP[label]
                    if i < len(bboxes):
                        box = bboxes[i]
                        xc, yc, nw, nh = convert_bbox(*box)
                        lines.append(f"{cls_id} {xc:.6f} {yc:.6f} {nw:.6f} {nh:.6f}")
                else:
                    unknown_labels.add(label)
            
            # Write to file (Overwrite)
            if lines:
                with open(label_file, 'w') as f:
                    f.write('\n'.join(lines))
                count += 1
                written_files.add(os.path.abspath(label_file))
                
    print(f"Finished! Regenerated {count} label files.")
    
    # Cleanup Orphans
    print("Cleaning up orphaned label files...")
    removed_count = 0
    for split in splits:
        label_dir = os.path.join(HAGRID_LABEL_ROOT, split)
        if not os.path.exists(label_dir): continue
        
        for f in os.listdir(label_dir):
            if f.endswith('.txt'):
                full_path = os.path.abspath(os.path.join(label_dir, f))
                if full_path not in written_files:
                    os.remove(full_path)
                    removed_count += 1
                    
    print(f"Removed {removed_count} orphaned label files.")
    
    if unknown_labels:
        print("\n⚠️  WARNING: Found unknown labels in JSONs (not in CLASS_MAP):")
        print(unknown_labels)
        print("Please update CLASS_MAP in test.py and this script if these are needed.")

if __name__ == "__main__":
    main()
