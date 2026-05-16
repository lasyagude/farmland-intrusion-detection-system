import os
import sys
import cv2
import numpy as np
import librosa
import tensorflow as tf
import tensorflow_hub as hub
from ultralytics import YOLO
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import pandas as pd

if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')


# ============================================================
# SETTINGS
# ============================================================
AUDIO_SR = 16000

CLASSES = ["cow", "goat", "elephant", "wildboar"]
VISION_CLASSES = ["cow", "elephant"]   # YOLO supports only these
AUDIO_ONLY_CLASSES = ["goat", "wildboar"]

TEST_DATASET = "test_samples"


# ============================================================
# LOAD MODELS
# ============================================================
print("\n📌 Loading YAMNet...")
yamnet = hub.load("https://tfhub.dev/google/yamnet/1")

print("📌 Loading quantized audio classifier (INT8)...")
interpreter = tf.lite.Interpreter(model_path="audio/animal_classifier_int8.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()[0]
output_details = interpreter.get_output_details()[0]

print("📌 Loading YOLOv8 ONNX model...")
video_model = YOLO("visual/best.onnx")

print("\n🚀 Running Integrated Accuracy Evaluation...\n")


# ============================================================
# AUDIO PREDICTION
# ============================================================
def predict_audio(path):
    try:
        audio, _ = librosa.load(path, sr=AUDIO_SR)
    except:
        return None, 0.0

    # Get embeddings using YAMNet
    _, embeddings, _ = yamnet(audio)
    emb = np.mean(embeddings, axis=0).astype(np.float32)

    # Quantize input (INT8 model expects quantized input)
    scale, zp = input_details["quantization"]
    emb_q = (emb / scale + zp).astype(np.int8).reshape(1, -1)

    interpreter.set_tensor(input_details['index'], emb_q)
    interpreter.invoke()

    raw_output = interpreter.get_tensor(output_details['index'])[0]
    out_scale, out_zp = output_details["quantization"]
    preds = (raw_output - out_zp) * out_scale

    class_id = np.argmax(preds)
    return CLASSES[class_id], float(preds[class_id])


# ============================================================
# VIDEO PREDICTION
# ============================================================
def predict_image(path):
    img = cv2.imread(path)
    if img is None:
        return None, 0.0

    results = video_model(img, verbose=False)[0]
    best_pred = None
    best_conf = 0.0

    for box in results.boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])

        # Make sure YOLO does not output unknown class index
        if cls_id >= len(VISION_CLASSES):
            continue  

        if conf > best_conf:
            best_pred = VISION_CLASSES[cls_id]
            best_conf = conf

    return best_pred, best_conf


# ============================================================
# FUSION LOGIC
# ============================================================
def fuse(audio_pred, audio_conf, img_pred, img_conf):

    # If class exists ONLY in audio model
    if audio_pred in AUDIO_ONLY_CLASSES:
        return audio_pred if audio_conf > 0.70 else "none"

    # If both detect same animal
    if audio_pred == img_pred and audio_conf > 0.60 and img_conf > 0.60:
        return img_pred

    # If YOLO is very confident (visual is dominant for cow/elephant)
    if img_pred in VISION_CLASSES and img_conf > 0.85:
        return img_pred

    # If audio is very strong and video failed
    if audio_conf > 0.85:
        return audio_pred

    return "none"


# ============================================================
# PROCESS TEST DATASET
# ============================================================
for cls in CLASSES:
    folder = os.path.join(TEST_DATASET, cls)

    if not os.path.exists(folder):
        print(f"⚠ Missing folder: {cls}")
        continue

    # Search in both 'audio' and 'images' subfolders
    for sub in ["audio", "images"]:
        subfolder = os.path.join(folder, sub)
        if not os.path.exists(subfolder):
            continue

        for file in os.listdir(subfolder):
            path = os.path.join(subfolder, file)

            audio_pred, audio_conf = None, 0
            img_pred, img_conf = None, 0

            if file.endswith(".wav"):
                audio_pred, audio_conf = predict_audio(path)
            elif file.endswith((".jpg", ".png", ".jpeg")):
                img_pred, img_conf = predict_image(path)
            else:
                continue

            final = fuse(audio_pred, audio_conf, img_pred, img_conf)

            print(f"🎯 {file} → TRUE={cls} | AUDIO={audio_pred} ({audio_conf:.2f}) | "
                  f"VIDEO={img_pred} ({img_conf:.2f}) → FINAL={final}")

            y_true.append(cls)
            y_pred.append(final)


# ============================================================
# RESULTS REPORT
# ============================================================
print("\n📊 FINAL PERFORMANCE SUMMARY")
print("-------------------------------------")

accuracy = accuracy_score(y_true, y_pred)
print(f"✔ Accuracy: {accuracy:.2f}\n")

print("✔ Classification Report:\n")
print(classification_report(y_true, y_pred, labels=CLASSES))

print("✔ Confusion Matrix:\n")
print(confusion_matrix(y_true, y_pred, labels=CLASSES))

pd.DataFrame({"True": y_true, "Predicted": y_pred}).to_csv("fusion_accuracy.csv", index=False)

print("\n📁 Results saved → fusion_accuracy.csv\n")
