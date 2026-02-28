# Face, Gesture, and Activity Game Control AI

This project implements various Computer Vision and Deep Learning techniques to interpret human body poses, face detection, hand gestures, and activities to create interactive, hands-free video game controllers. 

By utilizing a standard webcam, users can control games like Subway Surfers and racing simulation games through body movements and hand gestures, as well as perform real-time activity recognition.

## 🚀 Features & Functionalities

### 1. Body Pose Control (Subway Surfers Controller)
* **Custom Pose Data Collection (`capture_data.py`)**: A utility script to capture your own body pose landmarks via your webcam for states like `JUMP`, `DOWN`, `LEFT`, `RIGHT`, and `IDLE`.
* **Dataset Augmentation (`extract_pose_features.py`)**: Extracts and horizontally flips body pose landmarks from the MPII Human Pose Dataset to build a robust training set.
* **Model Training (`train_subway_model.py` / `pose_scripts/train_pose_model.py`)**: Trains a Custom Multi-Layer Perceptron (MLP) Classifier using PyTorch on the collected 132 landmark coordinates.
* **Real-time Game Control (`subway_control.py`)**: Analyzes continuous video feed to infer the user's stance using MediaPipe and the MLP model. High-confidence poses are mapped to keyboard presses (`UP`, `DOWN`, `LEFT`, `RIGHT`) using PyAutoGUI to navigate in-game.

### 2. Hand Gesture & Face Detection (Racing Games Controller)
* **Dual-Model Inference (`running.py`)**: Uses fine-tuned YOLO object detection models to concurrently detect user faces (`final_face_model.pt`) and recognize 18+ specific hand gestures (`final_face_gesture_model.pt`).
* **Gesture Game Controller (`game_control.py`)**: Leverages real-time object detection models to map specific hand gestures to keyboard inputs for driving video games. 
  * ✊ **Fist**: Gas / Accelerate (Right Arrow)
  * ✋ **Palm / Stop**: Brake / Reverse (Left Arrow)
* **Dataset Prep & Training (`test.py`)**: Automatically converts WIDER FACE and HaGRID annotations into YOLO format and trains the combined face and gesture detector.

### 3. General Activity Recognition
* **Model Training (`train_activity.py`, `mpii_utils.py`)**: Fine-tunes a ResNet50 neural network on human activity datasets (MPII) to categorize multi-frame actions (supports up to 397 classes).
* **Live Activity Monitoring (`webcam_activity.py`)**: Evaluates the live webcam stream, passing processed frames through the ResNet50 model to classify the user's ongoing physical activity and displays the live prediction confidence.

## 🛠️ Tech Stack & Technologies Used

This project is built directly utilizing modern Python data science and machine learning libraries:

* **Programming Language**: [Python](https://www.python.org/)
* **Deep Learning Frameworks**: 
  * [PyTorch](https://pytorch.org/) & [Torchvision](https://pytorch.org/vision/stable/index.html): Used for designing, training, and running the MLP (pose control) and ResNet50 (activity recognition) architectures.
  * [Ultralytics YOLO](https://docs.ultralytics.com/): Employed for rapid, state-of-the-art bounding-box inference for faces and complex hand gestures.
* **Computer Vision**:
  * [MediaPipe](https://developers.google.com/mediapipe): Powerful internal tools employed for highly accurate, sub-millisecond mapping of 33-point geometric human posture skeletal landmarks.
  * [OpenCV (`opencv-python`)](https://opencv.org/): Manages core video/image pre-processing, rendering bounding boxes, and hardware webcam I/O.
* **Automation & Control**: 
  * [PyAutoGUI](https://pyautogui.readthedocs.io/): Acts as the bridge connecting AI inference predictions to the OS, triggering autonomous simulated keyboard events.
* **Data Processing & Utilities**: 
  * [Pandas](https://pandas.pydata.org/) & [NumPy](https://numpy.org/): Core utilities for dataset manipulation and matrix evaluation.
  * [SciPy](https://scipy.org/): Decodes nested MATLAB (`.mat`) annotation structures from the MPII datasets.
  * [Pillow (PIL)](https://python-pillow.org/): Applied for high-level image augmentations and precision tensor conversions.
  * [PyYAML (`yaml`)](https://pyyaml.org/) & [tqdm](https://tqdm.github.io/): Managing model configuration files and training progress bars.

## 📁 Datasets Utilized

1. **WIDER FACE**: Used for face detection labels and images.
2. **HaGRID Hand Gesture Dataset**: Used for training gesture classes (`call`, `dislike`, `fist`, `palm`, `peace`, etc.).
3. **MPII Human Pose Dataset**: Used for activity classification metadata (`train_activity.py`) and pose landmark extraction (`extract_pose_features.py`).

## 🧠 Trained Model Artifacts

* `final_face_model.pt`: YOLO Face detector.
* `final_face_gesture_model.pt`: Combined YOLO Face and Hand-gesture detector.
* `subway_pose_model.pth`: PyTorch MLP mapping standard human poses (Jump/Down/Left/Right/Idle).
* `activity_model.pth` / `best_activity_model.pth`: PyTorch ResNet50 checkpoints for MPII activity.
* `yolo11n.pt` / `yolov8n.pt`: Original YOLO base weights used for fine-tuning.
* `activity_map.pkl` & `pose_label_map.pkl`: Serialized Python dictionaries linking class IDs to readable strings.

## ⚙️ Setup & Installation

**1. Virtual Environment:**
```bash
python -m venv .venv
.venv\Scripts\activate
```

**2. Dependencies:**
```bash
python -m pip install --upgrade pip
pip install ultralytics opencv-python mediapipe scipy pandas pillow pyyaml tqdm pyautogui
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```
*(If you do not have a CUDA-capable GPU, install the CPU-compatible version of PyTorch).*

## 🎮 How to Run

* **A. Run Live Face & Gesture Detection:**  
  `python running.py`
* **B. Play Racing Games via Hand Gestures:**  
  `python game_control.py`  
  *(Fist = Gas, Palm/Stop = Brake)*
* **C. Play Subway Surfers via Body Pose:**  
  `python subway_control.py --model subway_pose_model.pth`
* **D. Run Live Webcam Activity Recognition:**  
  `python webcam_activity.py --model best_activity_model.pth --map activity_map.pkl`
