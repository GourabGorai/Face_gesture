
import cv2
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import os
import argparse
import pyautogui
import time

# SPECIALIZED MODEL MAPPING
# 0: rope skipping -> UP
# 1: ballroom      -> DOWN
# 2: boxing        -> LEFT
# 3: fencing       -> RIGHT

ID_TO_ACTION = {
    0: 'up',
    1: 'down',
    2: 'left',
    3: 'right'
}

CONFIDENCE_THRESHOLD = 0.7
COOLDOWN = 0.5 

def load_model(model_path):
    print("Loading specialized model...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 4 Classes
    model = models.resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 4)
    
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print("Model loaded.")
    else:
        raise FileNotFoundError(f"Model not found at {model_path}")
    
    model = model.to(device)
    model.eval()
    
    return model, device

def main():
    parser = argparse.ArgumentParser(description='Subway Surfers Control Specialized')
    parser.add_argument('--model', type=str, default='subway_model.pth')
    args = parser.parse_args()
    
    base_path = r"d:\BragBoard-main\Face Detection"
    model_path = os.path.join(base_path, args.model)
    
    try:
        model, device = load_model(model_path)
    except Exception as e:
        print(f"Error: {e}")
        return

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    cap = cv2.VideoCapture(0)
    print("Starting Specialized Game Control. Press 'q' to quit.")
    print("Mapping: 0=UP(Rope), 1=DOWN(Ballroom), 2=LEFT(Boxing), 3=RIGHT(Fencing)")
    
    last_action_time = 0
    
    while True:
        ret, frame = cap.read()
        if not ret: break
        
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb_frame)
        input_tensor = transform(pil_img).unsqueeze(0).to(device)
        
        with torch.no_grad():
            outputs = model(input_tensor)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            conf, pred = torch.max(probs, 1)
            
        label_id = pred.item()
        confidence = conf.item()
        action = ID_TO_ACTION.get(label_id, "Unknown")
        
        # Overlay
        text = f"{action.upper()}: {confidence:.2f}"
        color = (0, 255, 0)
        if confidence < CONFIDENCE_THRESHOLD:
            color = (0, 0, 255) # Red if low confidence
            
        cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        
        # Control Logic
        current_time = time.time()
        if confidence > CONFIDENCE_THRESHOLD and (current_time - last_action_time > COOLDOWN):
            print(f"Action: {action.upper()}")
            pyautogui.press(action)
            last_action_time = current_time
            
            cv2.putText(frame, f"PRESS: {action.upper()}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

        cv2.imshow('Subway Surfers Control', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
