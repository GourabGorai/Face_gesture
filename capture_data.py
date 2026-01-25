
import cv2
import mediapipe as mp
import csv
import os
import sys
import time

# FORCE SYS PATH FIX (Just in case)
site_pkg = r"D:\BragBoard-main\face_system\Lib\site-packages"
if os.path.exists(site_pkg) and site_pkg not in sys.path:
     sys.path.insert(0, site_pkg)

# Explicit import patch
try:
    from mediapipe.python import solutions
    mp.solutions = solutions
except:
    pass

def capture_data():
    base_path = r"d:\BragBoard-main\Face Detection"
    output_csv = os.path.join(base_path, "my_pose_data.csv")
    
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
    
    # Define Classes
    # 0: JUMP, 1: DOWN, 2: LEFT, 3: RIGHT, 4: IDLE
    LABELS = {
        ord('j'): 0, # Jump
        ord('d'): 1, # Down
        ord('l'): 2, # Left
        ord('r'): 3, # Right
        ord('i'): 4  # Idle
    }
    
    LABEL_NAMES = {
        0: 'JUMP',
        1: 'DOWN', 
        2: 'LEFT',
        3: 'RIGHT',
        4: 'IDLE'
    }
    
    # Initialize CSV if needed
    if not os.path.exists(output_csv):
        header = ['label']
        for i in range(33):
            header.extend([f'x{i}', f'y{i}', f'z{i}', f'v{i}'])
        with open(output_csv, 'w', newline='') as f:
            csv.writer(f).writerow(header)
            
    # Load existing counts
    counts = {0:0, 1:0, 2:0, 3:0, 4:0}
    try:
        with open(output_csv, 'r') as f:
            reader = csv.reader(f)
            next(reader) # skip header
            for row in reader:
                if row:
                    l = int(row[0])
                    counts[l] = counts.get(l, 0) + 1
    except:
        pass

    cap = cv2.VideoCapture(0)
    
    print("=== DATA COLLECTION MODE ===")
    print("Hold these keys to record:")
    print("  'i' - IDLE  (Stand still)")
    print("  'j' - JUMP  (Perform jump/hands up)")
    print("  'd' - DOWN  (Crouch/Duck)")
    print("  'l' - LEFT  (Lean Left)")
    print("  'r' - RIGHT (Lean Right)")
    print("  'q' - QUIT")
    
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            
            # Mirror view for easier interaction
            frame = cv2.flip(frame, 1)
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            results = pose.process(image_rgb)
            
            status_text = "Press Key to Record"
            color = (255, 255, 255)
            
            if results.pose_landmarks:
                mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
                
                # Check for key press
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q'):
                    break
                
                if key in LABELS:
                    label_id = LABELS[key]
                    name = LABEL_NAMES[label_id]
                    
                    # Record Data
                    row = [label_id]
                    for landmark in results.pose_landmarks.landmark:
                        row.extend([landmark.x, landmark.y, landmark.z, landmark.visibility])
                        
                    with open(output_csv, 'a', newline='') as f:
                        csv.writer(f).writerow(row)
                        
                    counts[label_id] += 1
                    status_text = f"RECORDING: {name}"
                    color = (0, 0, 255)
            else:
                key = cv2.waitKey(1)
                if key == ord('q'): break

            # UI Overlay
            y = 30
            for lid, name in LABEL_NAMES.items():
                cnt = counts[lid]
                cv2.putText(frame, f"{name}: {cnt}", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                y += 25
                
            cv2.putText(frame, status_text, (300, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            
            cv2.imshow('Data Collector', frame)
            
    cap.release()
    cv2.destroyAllWindows()
    print("Data collection complete.")

if __name__ == "__main__":
    capture_data()
