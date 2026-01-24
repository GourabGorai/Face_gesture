
import cv2
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import pickle
import os
import argparse

def load_resources(model_path, map_path):
    print("Loading resources...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load Activity Map
    if not os.path.exists(map_path):
        raise FileNotFoundError(f"Activity map not found at {map_path}. Train the model first.")
    
    with open(map_path, 'rb') as f:
        activity_map = pickle.load(f)
    
    # Invert map for ID -> Name
    id_to_activity = {v: k for k, v in activity_map.items()}
    num_classes = len(activity_map)
    print(f"Loaded {num_classes} activities.")

    # Load Model
    model = models.resnet50(weights=None) # No need for weights desc download if loading state dict
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print("Model loaded successfully.")
    else:
        raise FileNotFoundError(f"Model not found at {model_path}")
    
    model = model.to(device)
    model.eval()
    
    return model, id_to_activity, device

def main():
    parser = argparse.ArgumentParser(description='Webcam Activity Recognition')
    parser.add_argument('--model', type=str, default='best_activity_model.pth', help='Path to model file')
    parser.add_argument('--map', type=str, default='activity_map.pkl', help='Path to activity map file')
    args = parser.parse_args()
    
    # Resource paths
    base_path = r"d:\BragBoard-main\Face Detection"
    model_path = os.path.join(base_path, args.model)
    map_path = os.path.join(base_path, args.map)
    
    # Fallback to standard model if best not found
    if not os.path.exists(model_path) and args.model == 'best_activity_model.pth':
         print("Best model not found, trying 'activity_model.pth'...")
         model_path = os.path.join(base_path, 'activity_model.pth')

    try:
        model, id_to_activity, device = load_resources(model_path, map_path)
    except Exception as e:
        print(f"Error loading resources: {e}")
        return

    # Transform
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    print("Starting webcam inference. Press 'q' to quit.")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Preprocess
        # CV2 is BGR, PIL is RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb_frame)
        
        input_tensor = transform(pil_img).unsqueeze(0).to(device)
        
        # Inference
        with torch.no_grad():
            outputs = model(input_tensor)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            conf, pred = torch.max(probs, 1)
            
        label_id = pred.item()
        confidence = conf.item()
        activity = id_to_activity.get(label_id, "Unknown")
        
        # Display
        text = f"{activity}: {confidence:.2f}"
        color = (0, 255, 0) if confidence > 0.5 else (0, 0, 255)
        
        cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        cv2.imshow('Activity Recognition', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
