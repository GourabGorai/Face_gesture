import torch
from ultralytics import YOLO
import os
import gc


def start_training():
    # ================= 1. GPU SETUP & VERIFICATION =================
    print("\n" + "=" * 50)
    print("🚀 SYSTEM CHECK")
    print("=" * 50)

    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"✅ GPU DETECTED: {gpu_name}")
        print(f"   VRAM: {gpu_mem:.2f} GB")
        device = 0
    else:
        print("❌ NO GPU DETECTED! Training will be extremely slow.")
        print("   If you have an NVIDIA GPU, check your PyTorch installation.")
        device = 'cpu'

    # ================= 2. DATASET CONFIGURATION =================
    # We create the YAML config file dynamically to ensure paths are perfect

    # ⚠️ IMPORTANT: Verify the 'Hand Gesture' image path below matches your folder

    yaml_content = f"""
    path: {os.getcwd()} # Project root

    train: 
      - Face Dataset/WIDER_train/WIDER_train/images
      - Hand Gesture/train_val_images  # <--- CHECK THIS FOLDER NAME IN YOUR PROJECT!

    val: 
      - Face Dataset/WIDER_val/WIDER_val/images
      - Hand Gesture/test_images       # <--- CHECK THIS FOLDER NAME!

    # Classes
    nc: 19
    names: ['face', 'call', 'dislike', 'fist', 'four', 'like', 'mute', 'ok', 'one', 'palm', 'peace', 'peace_inverted', 'rock', 'stop', 'three', 'three2', 'two_up', 'two_up_inverted', 'no_gesture']
    """

    # Write the config file
    with open("custom_config.yaml", "w") as f:
        f.write(yaml_content)
    print("✅ Configuration file 'custom_config.yaml' created.")

    # ================= 3. INITIALIZE MODEL =================
    # We use YOLOv11 Nano (n) for speed.
    # If you have a powerful GPU (RTX 3060+), you can try 'yolo11s.pt' (Small) or 'yolo11m.pt' (Medium)
    print("\n🧠 Loading Model (YOLOv11 Nano)...")
    model = YOLO("yolo11n.pt")

    # ================= 4. START TRAINING =================
    print("\n🔥 STARTING TRAINING ON GPU...")
    print("   (Press Ctrl+C ONCE to stop early and save progress)")

    try:
        results = model.train(
            data="custom_config.yaml",
            epochs=50,  # 50 epochs is a good balance
            imgsz=640,  # Standard image size
            batch=16,  # Set to 8 or 4 if you get "CUDA Out of Memory"
            device=device,  # Force GPU usage
            workers=4,  # Speed up data loading
            project="runs/detect",  # Output folder
            name="face_hand_gpu",  # Run name
            exist_ok=True,  # Overwrite if exists (optional)
            amp=True  # Automatic Mixed Precision (Faster on GPU)
        )
        print("\n✅ Training Finished Successfully!")

    except KeyboardInterrupt:
        print("\n⚠️ Training interrupted by user. Saving current progress...")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        if "CUDA out of memory" in str(e):
            print("👉 SOLUTION: Change 'batch=16' to 'batch=8' or 'batch=4' in the code.")

    # ================= 5. CLOSE & CLEANUP =================
    # Release GPU memory after training
    del model
    gc.collect()
    torch.cuda.empty_cache()
    print("🧹 GPU Memory Cleared.")


if __name__ == "__main__":
    start_training()