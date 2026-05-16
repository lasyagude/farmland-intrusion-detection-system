import cv2
import numpy as np
import sounddevice as sd
import tensorflow_hub as hub
import tensorflow as tf
import sys
from ultralytics import YOLO

if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

# ============================================================
# CONFIGURATION — tuned for edge deployment
# ============================================================
AUDIO_SR = 16000
AUDIO_DURATION = 1          # seconds
AUDIO_THRESHOLD = 0.65
VIDEO_THRESHOLD = 0.70

BEST_VIDEO_CONF = 0.85
BEST_AUDIO_CONF = 0.90

SKIP_FRAMES = 3            # run YOLO only every N frames

CLASSES = ["cow", "goat", "elephant", "wildboar"]
VISION_CLASSES = ["cow", "elephant"]  # YOLO model only supports these two

# ============================================================
# LOAD MODELS
# ============================================================
print("\nLoading YAMNet...")
yamnet = hub.load("https://tfhub.dev/google/yamnet/1")

print("Loading Quantized TFLite Audio Classifier...")
interpreter = tf.lite.Interpreter(model_path="audio/animal_classifier_int8.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()[0]
output_details = interpreter.get_output_details()[0]

print("Loading YOLOv8 ONNX Model...")
video_model = YOLO("visual/best.onnx")

# ============================================================
# AUDIO PROCESSING
# ============================================================
def record_audio():
    """Record raw mic audio"""
    audio = sd.rec(int(AUDIO_DURATION * AUDIO_SR), samplerate=AUDIO_SR,
                   channels=1, dtype="float32")
    sd.wait()
    audio = audio.flatten()

    # Noise floor check (ignore silence)
    if np.max(np.abs(audio)) < 0.02:
        return None
    return audio


def detect_audio():
    """Run YAMNet + TFLite classifier"""
    audio = record_audio()
    if audio is None:
        return None, 0.0

    _, embeddings, _ = yamnet(audio)
    emb = np.mean(embeddings, axis=0).astype(np.float32)

    # Quantize → INT8
    scale, zp = input_details["quantization"]
    emb_q = (emb / scale + zp).astype(np.int8).reshape(1, -1)

    interpreter.set_tensor(input_details['index'], emb_q)
    interpreter.invoke()

    # Dequantize output
    raw_output = interpreter.get_tensor(output_details['index'])[0]
    out_scale, out_zp = output_details["quantization"]
    output = (raw_output - out_zp) * out_scale

    idx = int(np.argmax(output))
    confidence = float(output[idx])
    return CLASSES[idx], confidence

# ============================================================
# VIDEO PROCESSING
# ============================================================
def detect_video(frame):
    """YOLO inference"""
    results = video_model(frame, verbose=False)[0]
    best_class, best_conf = None, 0.0

    for box in results.boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])

        if conf > best_conf and conf > VIDEO_THRESHOLD:
            # Fix: Use VISION_CLASSES for YOLO output mapping
            best_class = VISION_CLASSES[cls_id] if cls_id < len(VISION_CLASSES) else None
            best_conf = conf

    return best_class, best_conf


# ============================================================
# MAIN LOOP (SYNCED AUDIO + SKIPPED VIDEO)
# ============================================================
print("\n🚀 EDGE-OPTIMIZED LIVE FUSION SYSTEM STARTED\n")
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

frame_counter = 0

while True:
    frame_counter += 1
    ret, frame = cap.read()
    if not ret:
        print("⚠ Camera not available")
        continue

    # Skip frames to save compute power
    if frame_counter % SKIP_FRAMES != 0:
        cv2.imshow("Fusion Camera Feed", frame)
        if cv2.waitKey(1) == ord("q"):
            break
        continue

    # Run audio + video detections
    audio_animal, audio_conf = detect_audio()
    video_animal, video_conf = detect_video(frame)

    # Determine lighting condition (auto day/night mode)
    brightness = np.mean(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
    is_night = brightness < 60
    mode = "NIGHT" if is_night else "DAY"

    final_animal = "none"

    # ---------------- FUSION LOGIC ----------------
    if mode == "DAY":
        if audio_animal == video_animal and audio_conf > 0.7 and video_conf > 0.7:
            final_animal = audio_animal
        elif video_conf > BEST_VIDEO_CONF:
            final_animal = video_animal

    else:  # NIGHT MODE
        if audio_conf > BEST_AUDIO_CONF:
            final_animal = audio_animal

    # ---------------- DISPLAY + ALERT ----------------
    print("\n================ FUSION RESULT =================")
    print(f"Mode       : {mode}")
    print(f"Audio      : {audio_animal} ({audio_conf:.2f})")
    print(f"Video      : {video_animal} ({video_conf:.2f})")
    print(f"FINAL      : {final_animal.upper()}")
    print("================================================\n")

    cv2.putText(frame, f"FINAL: {final_animal}", (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

    if final_animal != "none":
        cv2.putText(frame, f"🚨 ALERT: {final_animal.upper()} 🚨", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)

    cv2.imshow("Fusion Camera Feed", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

print("\n🛑 SYSTEM STOPPED\n")
