# 🌾 Farmland Intrusion Detection System (FIDS)
### Multi-Modal Sensor Fusion for Edge AI

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Detection-green.svg)
![EdgeAI](https://img.shields.io/badge/Edge%20AI-Optimized-red.svg)

## 📌 Project Overview
FIDS is an intelligent monitoring system designed to protect farmlands from wild animal intrusions. It leverages **Audio-Visual Sensor Fusion** to detect and classify intruders (Cows, Elephants, Goats, and Wild Boars) in real-time. 

By combining computer vision with acoustic intelligence, the system maintains high accuracy even in challenging conditions where one sensor might fail (e.g., low light or visual obstructions).

---

## 🚀 Key Features
- **Multi-Modal Fusion**: Integrates **YOLOv8** (Visual) and **YAMNet** (Acoustic) for robust threat detection.
- **Dynamic Day/Night Switching**: Automatically shifts reliance between Camera and Microphone based on ambient light levels (Auto-NIGHT mode).
- **Edge Optimized**: Uses quantized **INT8 TFLite** models and **ONNX** runtimes for high-speed inference on low-power devices.
- **Intelligent Gating Logic**: A custom fusion algorithm that handles sensor confidence scores to minimize false positives.

---

## 🛠️ Technical Architecture

### 1. Vision System (YOLOv8)
- **Model**: Custom-trained YOLOv8 exported to ONNX.
- **Target Classes**: Cow, Elephant.
- **Optimization**: Runs on a skipped-frame logic (every 3rd frame) to conserve CPU/GPU resources.

### 2. Acoustic System (YAMNet + TFLite)
- **Feature Extraction**: Uses Google's **YAMNet** to generate high-dimensional audio embeddings.
- **Classifier**: A custom-trained dense neural network quantized to **INT8** for ultra-fast audio classification.
- **Target Classes**: Cow, Goat, Elephant, Wild Boar.

### 3. Fusion Logic (The "Brain")
- **Day Mode**: Priority is given to visual confirmation or high-confidence agreement between both sensors.
- **Night Mode**: Relying primarily on the Acoustic sensor when visual visibility drops below a set brightness threshold.

---

## 📂 Project Structure
```text
├── integrated.py      # Main Live Fusion System (Camera + Mic)
├── acu.py             # Performance Evaluation & Accuracy Metrics
├── debug.py           # Dataset Integrity & Path Verification
├── audio/             # TFLite Audio Classifier Models
├── visual/            # YOLOv8 ONNX Models
└── test_samples/      # Structured Test Dataset (Audio & Images)
```

---

## ⚙️ Installation & Usage

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/farmland-intrusion.git
   cd farmland-intrusion
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Live System:**
   ```bash
   python integrated.py
   ```

4. **Run Accuracy Evaluation:**
   ```bash
   python acu.py
   ```

---

## 👨‍💻 For Recruiters
This project demonstrates proficiency in:
- **Computer Vision**: Object detection and ONNX optimization.
- **Digital Signal Processing**: Audio feature extraction and embedding analysis.
- **Sensor Fusion**: Implementing decision-level fusion algorithms.
- **Software Engineering**: Writing clean, modular, and cross-platform compatible Python code.

---

## 📄 License
Distributed under the MIT License. See `LICENSE` for more information.
