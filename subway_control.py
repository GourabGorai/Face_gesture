
import cv2
import mediapipe as mp
import torch
import torch.nn as nn
import pickle
import os
import pyautogui
import time
import numpy as np
import argparse

# PoseClassifier definition must match training
class PoseClassifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(PoseClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.2)
        
        self.fc2 = nn.Linear(128, 64)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(0.2)
        
        self.fc3 = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.dropout2(x)
        x = self.fc3(x)
        return x

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='subway_pose_model.pth')
    args = parser.parse_args()

    base_path = r"d:\BragBoard-main\Face Detection"
    model_path = os.path.join(base_path, args.model)
    map_path = os.path.join(base_path, "pose_label_map.pkl")
    
    if not os.path.exists(model_path):
        print(f"Model {model_path} not found. Train it first!")
        return

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load Label Map
    try:
        with open(map_path, 'rb') as f:
            label_map = pickle.load(f)
    except:
        label_map = {0:'JUMP', 1:'DOWN', 2:'LEFT', 3:'RIGHT', 4:'IDLE'}
    
    print(f"Label Map: {label_map}")

    # Load Model
    model = PoseClassifier(132, 5)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    # MediaPipe
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
    
    cap = cv2.VideoCapture(0)
    
    print("Starting Pose Control. Press 'q' to quit.")
    
    last_action_time = 0
    COOLDOWN = 0.5
    
    # Key Mapping
    KEY_MAP = {
        'JUMP': 'up',
        'DOWN': 'down',
        'LEFT': 'left',
        'RIGHT': 'right',
        'IDLE': None
    }

    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            
            # Flip horizontally for mirror effect
            frame = cv2.flip(frame, 1)
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            results = pose.process(image_rgb)
            
            action_text = "Waiting..."
            color = (255, 255, 255)
            
            if results.pose_landmarks:
                # Draw skeleton
                mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
                
                # Extract features
                row = []
                for landmark in results.pose_landmarks.landmark:
                    row.extend([landmark.x, landmark.y, landmark.z, landmark.visibility])
                
                # Inference
                input_tensor = torch.FloatTensor(row).unsqueeze(0).to(device)
                with torch.no_grad():
                    outputs = model(input_tensor)
                    probs = torch.nn.functional.softmax(outputs, dim=1)
                    conf, pred = torch.max(probs, 1)
                
                label_id = pred.item()
                confidence = conf.item()
                action_name = label_map.get(label_id, "Unknown")
                
                # Display
                if confidence > 0.7:
                    color = (0, 255, 0)
                    action_text = f"{action_name} ({confidence:.2f})"
                    
                    # Trigger Key
                    key = KEY_MAP.get(action_name)
                    current_time = time.time()
                    
                    if key and (current_time - last_action_time > COOLDOWN):
                        print(f"Action: {action_name} -> Key: {key}")
                        pyautogui.press(key)
                        last_action_time = current_time
                        
                        cv2.putText(frame, f"PRESS: {key.upper()}", (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)
                else:
                    color = (0, 255, 255) # Low confidence
                    action_text = f"{action_name}? ({confidence:.2f})"
            
            # Status Bar
            cv2.rectangle(frame, (0,0), (640, 60), (0,0,0), -1)
            cv2.putText(frame, action_text, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            
            cv2.imshow('Subway Pose Control', frame)
            if cv2.waitKey(5) & 0xFF == ord('q'):
                break
                
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
