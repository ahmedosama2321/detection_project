# Pen Type Detection Using YOLO

A real-time computer vision project that uses a custom-trained YOLO model to detect and identify different types of pens through a webcam.

## 🚀 Features

- Real-time pen detection using a webcam.
- Identifies pen types based on the classes learned during model training.
- Displays bounding boxes around detected pens.
- Shows predicted class labels and confidence scores, depending on the implementation.
- Uses a custom-trained YOLO model for object detection.

## 🛠️ Technologies Used

- Python
- Ultralytics YOLO
- PyTorch
- OpenCV

## 📁 Project Structure

```text
detection_project/
├── models/
│   └── stationery_yolo_final.pt
├── webcam.py
├── requirements.txt
├── .gitignore
└── README.md
```

## ⚙️ Installation

### 1. Clone the Repository

```bash
git clone https://github.com/ahmedosama2321/detection_project.git
cd detection_project
```

### 2. Create a Virtual Environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
python -m pip install --upgrade pip
python -m pip install ultralytics opencv-python
```

Alternatively, install the dependencies from `requirements.txt`:

```bash
python -m pip install -r requirements.txt
```

## ▶️ Run the Project

Make sure the trained model is available at:

```text
models/stationery_yolo_final.pt
```

Start the webcam detection application:

```bash
python webcam.py
```

The webcam will display the model's predictions for the pen types it was trained to recognize.

## 🧠 Model

The project uses custom-trained YOLO weights stored in `models/stationery_yolo_final.pt`.

The model predicts object classes learned from its training dataset. Its ability to distinguish specific pen types depends on the training data, class labels, and model performance.

## 📋 Requirements

- Python environment compatible with the installed packages.
- Webcam.
- Trained model weights.
- Required Python dependencies.

## 👨‍💻 Author

**Ahmed Osama**

GitHub: [@ahmedosama2321](https://github.com/ahmedosama2321)

## 📄 License

For educational and portfolio purposes. Verify the applicable licenses for the model, dataset, and dependencies before redistribution.
