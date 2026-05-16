# 🌾 Farmland Intrusion Detection System (FIDS)
### Multi-Modal Sensor Fusion for Real-Time Animal Detection

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![TensorFlow](https://img.smanship.org/badge/TensorFlow-2.16-orange.svg)
![YOLOv8](https://img.shields.io/badge/YOLOv8-ONNX-green.svg)
![EdgeAI](https://img.shields.io/badge/Edge%20AI-Quantized-red.svg)

## 📌 Project Overview
The Farmland Intrusion Detection System (FIDS) is an intelligent monitoring solution designed to protect agricultural zones from animal-related damage. It employs a **decision-level fusion** architecture, combining computer vision (YOLOv8) and acoustic fingerprinting (YAMNet + TFLite) to identify intruders like elephants, wild boars, and livestock with high reliability.

By merging visual and acoustic signals, the system maintains high accuracy even in challenging conditions such as low light or visual occlusion.

---

## 🚀 Key Features
- **Multi-Modal Sensor Fusion**: Integrates **YOLOv8** (Visual) and **YAMNet** (Acoustic) for robust threat detection and reduced false alarms.
- **Adaptive Day/Night Logic**: Automatically adjusts detection sensitivity and sensor weighting based on real-time ambient light analysis.
- **Edge-Optimized Inference**: Utilizes quantized **INT8 TFLite** models and **ONNX** runtimes for high-speed performance on low-power edge hardware.
- **Intelligent Gating Logic**: A custom fusion algorithm that reconciles sensor confidence scores to maximize detection accuracy.

---

## 🛠️ Technical Architecture

### 1. Vision System (YOLOv8)
- **Model**: Custom-trained YOLOv8 exported to **ONNX**.
- **Role**: Provides spatial detection and high-confidence identification of large intruders (Cows, Elephants) during daylight.
- **Optimization**: Implements frame-skip logic to conserve computational resources on edge devices.

### 2. Acoustic System (YAMNet + TFLite)
- **Feature Extraction**: Uses Google's **YAMNet** to generate high-dimensional audio embeddings.
- **Classifier**: A custom-trained neural network quantized to **INT8** for ultra-fast audio classification.
- **Target Classes**: Cow, Goat, Elephant, Wild Boar.

### 3. Fusion Decision Matrix (The "Brain")
The system uses a custom gating algorithm to reconcile inputs:
- **Agreement Mode**: Triggered when both sensors confirm a class with >0.60 confidence.
- **Visual Dominance (Day)**: High-confidence YOLO detections override inconclusive audio signals.
- **Acoustic Dominance (Night)**: When brightness falls below a set threshold, the system automatically shifts priority to acoustic signatures.

---

## 📂 Repository Structure
```text
├── integrated.py      # Main Live Fusion System (Camera + Mic)
├── acu.py             # Performance Evaluation & Accuracy Metrics
├── debug.py           # Dataset Integrity & Path Verification Utility
├── audio/             # TFLite Audio Classifier Models
├── visual/            # YOLOv8 ONNX Models
└── test_samples/      # Structured Test Dataset (Audio & Images)
```

---

## ⚙️ Installation & Usage

1. **Clone the repository:**
   ```bash
   git clone https://github.com/lasyagude/farmland-intrusion-detection-system.git
   cd farmland-intrusion-detection-system
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

## 📈 Future Enhancements
- **LoRaWAN Integration**: For long-range alert transmission in remote agricultural areas.
- **Solar Power Management**: Implementing ultra-low-power modes for energy harvesting.
- **Thermal Imaging Support**: Enhancing Night Mode with FLIR sensor integration.

---

## 📄 License
Distributed under the MIT License.
