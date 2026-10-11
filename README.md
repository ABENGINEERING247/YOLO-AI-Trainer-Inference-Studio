# 🤖 YOLO AI Model Trainer & Media Inference Studio

An all-in-one **Streamlit application** for training custom **Ultralytics YOLO** object detection models from uploaded datasets, downloading trained model weights (`.pt`), and testing your models in real time on both **images (pictures)** and **videos**.

Developed by **Engr. Bilal Mehmood** (Certified Generative and Agentic AI Application Developer).

---

## 🌟 Key Features

* **🚀 Custom Dataset Training:** Upload standard YOLO-formatted datasets packaged inside a `.zip` file (including `data.yaml`, images, and labels). Configure training epochs, batch size, image resolution, and base model variants (`yolov8n`, `yolov8s`, `yolov8m`).
* **📦 Model Weight Exporter:** Automatically packages your trained `best.pt` weights into a downloadable `.zip` file.
* **🎥 Media Inference Studio:**
* **Image Testing:** Upload pictures (`.jpg`, `.png`) to run instant bounding box predictions with customizable confidence thresholds.
* **Video Processing:** Upload video files (`.mp4`, `.avi`, `.mov`) to perform frame-by-frame object detection, preview the processed video live, and download the resulting output file.


* **⚡ Seamless Workflow:** Transition instantly from training your model to testing it on live media within a unified tabbed interface.

---

## 🚀 Quick Start Guide

### 1. Prerequisites

Ensure you have **Python 3.8+** installed on your system.

### 2. Installation

Clone or download the project folder, then install the required dependencies:

```bash
pip install -r requirements.txt

```

### 3. Running the App

Launch the Streamlit web application:

```bash
streamlit run app_yolo_complete.py

```

The app will open automatically in your browser at `http://localhost:8501`.

---

## 🛠 Project Structure

```text
yolo-ai-studio/
│
├── app_yolo_complete.py   # Main Streamlit Application (Trainer & Inference Studio)
├── README.md              # Project Documentation
└── requirements.txt       # Python Dependencies

```

---

## 📦 `requirements.txt`

```text
streamlit>=1.30.0
ultralytics>=8.0.0
opencv-python-headless>=4.8.0
torch>=2.0.0
torchvision>=0.15.0
pandas>=2.0.0
numpy>=1.24.0
Pillow>=9.0.0

```

---

## 📁 Dataset Format Requirements (`.zip`)

Your uploaded dataset zip folder must follow the standard **Ultralytics YOLO directory layout**:

```text
dataset.zip
│
├── data.yaml           # YAML file specifying train/val paths and class names
├── train/
│   ├── images/         # Training images (.jpg/.png)
│   └── labels/         # Corresponding YOLO bounding box text files (.txt)
└── val/
    ├── images/         # Validation images
    └── labels/         # Corresponding validation label files

```

---

## 👨‍💻 Author & Developer

* **Developer:** **Engr. Bilal Mehmood**
* **Certification:** Certified Generative & Agentic AI Application Developer

---

## 🤝 Contributing

Contributions, bug reports, and feature requests are welcome! Feel free to fork the repository and open a pull request.
