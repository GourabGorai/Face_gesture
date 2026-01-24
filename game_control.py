import cv2
import pyautogui
from ultralytics import YOLO

# --- CONFIGURATION ---
MODEL_PATH = "final_face_gesture_model.pt"
CONFIDENCE_THRESHOLD = 0.5

# Gesture Mapping
# Look at running.py CLASS_MAP:
# "fist": 3, "palm": 9, "stop": 13
GESTURE_GAS = 3    # Fist
GESTURE_BRAKE_1 = 9  # Palm
GESTURE_BRAKE_2 = 13 # Stop

# Game Keys
KEY_GAS = 'right'
KEY_BRAKE = 'left'

# State tracking to avoid spamming key presses
current_action = "none" # "gas", "brake", "none"

def main():
    global current_action
    
    # 1. Load Model
    print(f"Loading gesture model: {MODEL_PATH}...")
    try:
        model = YOLO(MODEL_PATH)
        print("✅ Model loaded!")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return

    # 2. Setup Camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Error: Could not open webcam.")
        return

    print("--- RACING CONTROLLER STARTED ---")
    print("✊ FIST  = GAS (Right Arrow)")
    print("✋ PALM  = BRAKE (Left Arrow)")
    print("Press 'q' to quit.")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Use same inference logic as running.py (classes 1-18 for gestures)
            results = model(frame, conf=CONFIDENCE_THRESHOLD, classes=list(range(1, 19)), verbose=False)
            
            detected_gesture_id = None
            detected_gesture_name = ""

            # Standard detection processing
            if len(results[0].boxes) > 0:
                # Pick the highest confidence detection
                box = results[0].boxes[0]
                detected_gesture_id = int(box.cls[0])
                detected_gesture_name = model.names[detected_gesture_id]
            
            # --- CONTROL LOGIC ---
            new_action = "none"

            if detected_gesture_id == GESTURE_GAS:
                new_action = "gas"
            elif detected_gesture_id == GESTURE_BRAKE_1 or detected_gesture_id == GESTURE_BRAKE_2:
                new_action = "brake"
            
            # Apply Inputs (State Machine)
            if new_action != current_action:
                # Release previous keys
                if current_action == "gas":
                    pyautogui.keyUp(KEY_GAS)
                    print("Released Gas")
                elif current_action == "brake":
                    pyautogui.keyUp(KEY_BRAKE)
                    print("Released Brake")
                
                # Press new keys
                if new_action == "gas":
                    pyautogui.keyDown(KEY_GAS)
                    print(">> GAS (Fist)")
                elif new_action == "brake":
                    pyautogui.keyDown(KEY_BRAKE)
                    print(">> BRAKE (Palm)")
                
                current_action = new_action

            # --- VISUALIZATION ---
            # Draw detections
            annotated_frame = results[0].plot()
            
            # Overlay status text
            status_text = f"ACTION: {current_action.upper()}"
            color = (0, 255, 0) if current_action == "none" else (0, 0, 255)
            cv2.putText(annotated_frame, status_text, (50, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

            cv2.imshow("Gesture Game Control", annotated_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    except KeyboardInterrupt:
        pass
    finally:
        # Cleanup: Ensure all keys are released
        pyautogui.keyUp(KEY_GAS)
        pyautogui.keyUp(KEY_BRAKE)
        cap.release()
        cv2.destroyAllWindows()
        print("Stopped.")

if __name__ == "__main__":
    main()
