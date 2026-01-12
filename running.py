import cv2
import os
from ultralytics import YOLO

# --- CONFIGURATION ---
# This must match the name defined in your training script
# --- CONFIGURATION ---
# Model Paths
MODEL_FACE_PATH = "final_face_model.pt"
MODEL_HAND_PATH = "final_face_gesture_model.pt"

# CLASS MAPPING (Must match training)
CLASS_MAP = {
    "face": 0,
    "call": 1, "dislike": 2, "fist": 3, "four": 4, "like": 5,
    "mute": 6, "ok": 7, "one": 8, "palm": 9, "peace": 10,
    "peace_inverted": 11, "rock": 12, "stop": 13, "three": 14,
    "three2": 15, "two_up": 16, "two_up_inverted": 17, "no_gesture": 18
}


def test_model():
    # 1. Check if models exist
    if not os.path.exists(MODEL_FACE_PATH):
        print(f"❌ Error: Face Model '{MODEL_FACE_PATH}' not found.")
        return
    if not os.path.exists(MODEL_HAND_PATH):
        print(f"❌ Error: Hand Model '{MODEL_HAND_PATH}' not found.")
        return

    # 2. Load the trained models
    print("Loading models...")
    try:
        model_face = YOLO(MODEL_FACE_PATH)
        model_hand = YOLO(MODEL_HAND_PATH)
        print("✅ Models loaded successfully!")
    except Exception as e:
        print(f"❌ Failed to load models: {e}")
        return

    # 3. Open Webcam (Index 0 is usually the default camera)
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("❌ Error: Could not open webcam.")
        return

    print("✅ Webcam started! Press 'q' to exit.")
    
    print("\n--- DETECTING ---")
    print("Faces: using final_face_model.pt")
    print("Hands: using final_face_gesture_model.pt (Classes 1-18)")
    print("-----------------\n")

    # 4. Inference Loop
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        # --- Face Detection ---
        # Run inference for faces (conf=0.5)
        # We assume this model is good at faces.
        results_face = model_face(frame, conf=0.5, verbose=False)

        # --- Hand Gesture Detection ---
        # Run inference for hands, explicitly filtering for gesture classes (1-18)
        # This ignores class 0 (face) from this model to avoid conflict/suppression issues.
        results_hand = model_hand(frame, conf=0.5, classes=list(range(1, 19)), verbose=False)

        # --- Visualize & Combine ---
        # 1. Plot face detections on the original frame
        annotated_frame = results_face[0].plot()
        
        # 2. Plot hand detections on top of the face-annotated frame
        # We pass 'img=annotated_frame' so it draws on the existing array
        annotated_frame = results_hand[0].plot(img=annotated_frame)

        # Display the output
        cv2.imshow('Face & Gesture Test (Dual Model)', annotated_frame)

        # Press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 5. Cleanup
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    test_model()