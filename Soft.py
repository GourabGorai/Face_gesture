import os
import yaml  # pip install PyYAML


def find_image_directory(base_path, dataset_name):
    print(f"🔎 Scanning '{dataset_name}' for images...")

    # Walk through the directory to find where the .jpgs are hiding
    for root, dirs, files in os.walk(base_path):
        # fast check: if we see a .jpg, we found the image folder
        if any(f.lower().endswith('.jpg') for f in files):
            print(f"   ✅ Found images in: {root}")
            return root

    print(f"   ❌ NO IMAGES FOUND in {base_path}!")
    return None


def fix_configuration():
    base_dir = os.getcwd()

    # 1. FIND FACE IMAGES
    # We know from your screenshot they are nested
    face_root = os.path.join(base_dir, "Face Dataset")
    face_train_dir = find_image_directory(os.path.join(face_root, "WIDER_train"), "Face Train")
    face_val_dir = find_image_directory(os.path.join(face_root, "WIDER_val"), "Face Val")

    # 2. FIND HAND IMAGES
    hand_root = os.path.join(base_dir, "Hand Gesture")
    hand_img_dir = find_image_directory(hand_root, "Hand Gesture")

    # 3. VERIFY RESULTS
    missing_data = []
    if not face_train_dir: missing_data.append("Face Training Images")
    if not hand_img_dir: missing_data.append("Hand Gesture Images (HaGRID)")

    if missing_data:
        print("\n" + "!" * 50)
        print("CRITICAL ERROR: MISSING DATASETS")
        print("!" * 50)
        for item in missing_data:
            print(f"❌ Could not find: {item}")

        if "Hand Gesture Images (HaGRID)" in missing_data:
            print("\n👉 DIAGNOSIS: You have the 'ann_' (annotation) folders, but NOT the images.")
            print("   You must download the HaGRID 'subsample' (or full) IMAGE dataset.")
            print("   The folders you have (ann_subsample) only contain text files, not photos.")
        return

    # 4. GENERATE NEW CONFIG
    # If we found everything, we write the valid config file automatically
    print("\n✅ All datasets found! Generating 'custom_config.yaml'...")

    yaml_content = {
        'path': base_dir,
        'train': [face_train_dir, hand_img_dir],
        'val': [face_val_dir, hand_img_dir],  # Using same hand images for val if no separate test set found
        'nc': 19,
        'names': ['face', 'call', 'dislike', 'fist', 'four', 'like', 'mute', 'ok', 'one', 'palm', 'peace',
                  'peace_inverted', 'rock', 'stop', 'three', 'three2', 'two_up', 'two_up_inverted', 'no_gesture']
    }

    with open("custom_config.yaml", "w") as f:
        yaml.dump(yaml_content, f, default_flow_style=False)

    print("🎉 'custom_config.yaml' has been fixed!")
    print("👉 You can now run 'train_gpu.py' again.")


if __name__ == "__main__":
    fix_configuration()