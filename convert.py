import os
import cv2
import json
from tqdm import tqdm

# ================= 1. CONFIGURATION =================
# ✅ UPDATED PATHS BASED ON YOUR SCREENSHOTS

CONFIG = {
    # --- WIDER FACE SETTINGS ---
    # Screenshot shows nested structure: Face Dataset/WIDER_train/WIDER_train/images
    "WIDER_IMAGES": "Face Dataset/WIDER_train/WIDER_train/images",

    # Screenshot shows annotations inside 'wider_face_split'
    "WIDER_ANNOTATIONS": "Face Dataset/wider_face_annotations/wider_face_split/wider_face_train_bbx_gt.txt",

    # We will save labels next to the images folder for cleanliness
    "WIDER_OUTPUT": "Face Dataset/WIDER_train/WIDER_train/labels",

    # --- HaGRID SETTINGS ---
    "HAGRID_ANNOTATIONS_DIR": "Hand Gesture/ann_train_val",
    "HAGRID_OUTPUT": "Hand Gesture/labels",
}

# Unified Class Map
# 0 = Face, 1-18 = Hand Gestures
CLASS_MAP = {
    "face": 0,
    "call": 1, "dislike": 2, "fist": 3, "four": 4, "like": 5,
    "mute": 6, "ok": 7, "one": 8, "palm": 9, "peace": 10,
    "peace_inverted": 11, "rock": 12, "stop": 13, "three": 14,
    "three2": 15, "two_up": 16, "two_up_inverted": 17, "no_gesture": 18
}


# ================= 2. UTILITY FUNCTIONS =================

def convert_bbox(x, y, w, h, img_w=1, img_h=1, mode='corner'):
    """
    Converts bbox to YOLO format (center_x, center_y, width, height) normalized 0-1.
    """
    if mode == 'corner':
        # WIDER FACE: Raw pixel values -> Normalize -> Center
        x_center = (x + w / 2) / img_w
        y_center = (y + h / 2) / img_h
        norm_w = w / img_w
        norm_h = h / img_h
    elif mode == 'hagrid':
        # HaGRID: Already normalized 0-1 -> Just needs centering
        x_center = x + (w / 2)
        y_center = y + (h / 2)
        norm_w = w
        norm_h = h

    return (
        max(0, min(1, x_center)),
        max(0, min(1, y_center)),
        max(0, min(1, norm_w)),
        max(0, min(1, norm_h))
    )


# ================= 3. WIDER FACE PROCESSOR =================

def process_wider():
    print(f"\n🚀 [1/2] Processing WIDER FACE Dataset...")

    # DEBUG: Print paths being checked
    print(f"   📂 Checking for images in: {CONFIG['WIDER_IMAGES']}")
    print(f"   📄 Checking for annotations in: {CONFIG['WIDER_ANNOTATIONS']}")

    if not os.path.exists(CONFIG["WIDER_ANNOTATIONS"]):
        print(f"❌ Error: Annotation file NOT found.")
        return

    if not os.path.exists(CONFIG["WIDER_IMAGES"]):
        print(f"❌ Error: Images folder NOT found. Please verify the nested 'WIDER_train/WIDER_train/images' path.")
        return

    os.makedirs(CONFIG["WIDER_OUTPUT"], exist_ok=True)

    with open(CONFIG["WIDER_ANNOTATIONS"], 'r') as f:
        lines = f.readlines()

    i = 0
    count = 0
    pbar = tqdm(total=len(lines), desc="WIDER Faces")

    while i < len(lines):
        file_name = lines[i].strip()
        i += 1
        pbar.update(1)

        if not file_name.endswith('.jpg'): continue

        try:
            num_boxes = int(lines[i].strip())
            i += 1
            pbar.update(1)
        except ValueError:
            continue

        # Read Image for Dimensions
        img_path = os.path.join(CONFIG["WIDER_IMAGES"], file_name)

        if not os.path.exists(img_path):
            # If image isn't found, skip its boxes
            i += num_boxes
            pbar.update(num_boxes)
            continue

        img = cv2.imread(img_path)
        if img is None:
            i += num_boxes
            pbar.update(num_boxes)
            continue

        h_img, w_img, _ = img.shape

        # Prepare Output Path
        label_subdir = os.path.dirname(file_name)
        label_name = os.path.splitext(os.path.basename(file_name))[0] + ".txt"
        save_dir = os.path.join(CONFIG["WIDER_OUTPUT"], label_subdir)
        os.makedirs(save_dir, exist_ok=True)

        yolo_data = []
        for _ in range(num_boxes):
            box = lines[i].strip().split()
            i += 1
            pbar.update(1)

            x1, y1, w, h = map(int, box[0:4])

            xc, yc, nw, nh = convert_bbox(x1, y1, w, h, w_img, h_img, mode='corner')
            yolo_data.append(f"{CLASS_MAP['face']} {xc:.6f} {yc:.6f} {nw:.6f} {nh:.6f}")

        if yolo_data:
            with open(os.path.join(save_dir, label_name), 'w') as f:
                f.write('\n'.join(yolo_data))
            count += 1

    pbar.close()
    print(f"✅ WIDER FACE Complete! Generated labels for {count} images.")


# ================= 4. HaGRID PROCESSOR =================

def process_hagrid():
    print(f"\n🚀 [2/2] Processing HaGRID Dataset...")

    if not os.path.exists(CONFIG["HAGRID_ANNOTATIONS_DIR"]):
        print(f"❌ Error: HaGRID folder missing at {CONFIG['HAGRID_ANNOTATIONS_DIR']}")
        return

    os.makedirs(CONFIG["HAGRID_OUTPUT"], exist_ok=True)

    json_files = [f for f in os.listdir(CONFIG["HAGRID_ANNOTATIONS_DIR"]) if f.endswith('.json')]
    if not json_files:
        print("❌ No JSON files found for HaGRID.")
        return

    count = 0
    for json_file in json_files:
        path = os.path.join(CONFIG["HAGRID_ANNOTATIONS_DIR"], json_file)
        with open(path, 'r') as f:
            data = json.load(f)

        for img_id, content in tqdm(data.items(), desc=f"Reading {json_file}"):
            bboxes = content.get('bboxes', [])
            labels = content.get('labels', [])

            yolo_data = []
            for j in range(len(bboxes)):
                label = labels[j]
                if label in CLASS_MAP:
                    class_id = CLASS_MAP[label]
                    x, y, w, h = bboxes[j]

                    xc, yc, nw, nh = convert_bbox(x, y, w, h, mode='hagrid')
                    yolo_data.append(f"{class_id} {xc:.6f} {yc:.6f} {nw:.6f} {nh:.6f}")

            if yolo_data:
                # Save as {image_id}.txt
                out_path = os.path.join(CONFIG["HAGRID_OUTPUT"], f"{img_id}.txt")
                with open(out_path, 'w') as f:
                    f.write('\n'.join(yolo_data))
                count += 1

    print(f"✅ HaGRID Complete! Generated labels for {count} images.")


# ================= 5. MAIN EXECUTION =================

if __name__ == "__main__":
    try:
        process_wider()
        process_hagrid()
        print("\n🎉 ALL DATA PREPARED SUCCESSFULLY!")
    except KeyboardInterrupt:
        print("\n⚠️ Process interrupted by user.")