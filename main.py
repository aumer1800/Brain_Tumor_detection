# main.py
import os
import numpy as np
import uuid
from flask import Flask, request, jsonify, send_from_directory, make_response
from flask_cors import CORS
from tensorflow.keras.models import load_model

from utils.preprocessing import preprocess_image
from utils.gradcam import get_gradcam_heatmap, save_and_overlay_heatmap

# =========================
# APP SETUP
# =========================
app = Flask(__name__)

CORS(app, resources={r"/*": {
    "origins": "*",
    "methods": ["GET", "POST", "OPTIONS"],
    "allow_headers": ["Content-Type", "Authorization"]
}})

# =========================
# PATHS
# =========================
BASE_DIR = os.path.abspath(os.path.dirname(__file__))

UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
OUTPUT_FOLDER = os.path.join(BASE_DIR, "outputs")
FRONTEND_FOLDER = os.path.join(BASE_DIR, "frontend")

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
os.makedirs(FRONTEND_FOLDER, exist_ok=True)

MODEL_PATH = os.path.abspath(os.path.join(BASE_DIR, "..", "model", "best_model.h5"))
MATRIX_PATH = os.path.abspath(os.path.join(BASE_DIR, "..", "model", "confusion_matrix.npy"))

print("Loading model:", MODEL_PATH)

model = load_model(MODEL_PATH, compile=False)
_ = model.predict(np.zeros((1, 224, 224, 3)), verbose=0)

CLASS_NAMES = ["glioma", "meningioma", "notumor", "pituitary"]

# =========================
# SERVE FRONTEND
# =========================
@app.route("/")
def index():
    return send_from_directory(FRONTEND_FOLDER, "index.html")

@app.route("/<path:filename>")
def static_files(filename):
    return send_from_directory(FRONTEND_FOLDER, filename)

# =========================
# METRICS (SAFE VERSION)
# =========================
def fetch_real_validation_metrics():

    if not os.path.exists(MATRIX_PATH):
        return {
            "accuracy": 0,
            "precision": 0,
            "recall": 0,
            "specificity": 0,
            "f1_score": 0
        }

    try:
        cm = np.load(MATRIX_PATH)

        tp = np.diag(cm)
        fp = cm.sum(axis=0) - tp
        fn = cm.sum(axis=1) - tp
        tn = cm.sum() - (fp + fn + tp)

        accuracy = np.sum(tp) / np.sum(cm)
        precision = np.mean(tp / (tp + fp + 1e-7))
        recall = np.mean(tp / (tp + fn + 1e-7))
        specificity = np.mean(tn / (tn + fp + 1e-7))

        f1 = np.mean(
            2 * precision * recall / (precision + recall + 1e-7)
        )

        return {
            "accuracy": float(round(accuracy, 4)),
            "precision": float(round(precision, 4)),
            "recall": float(round(recall, 4)),
            "specificity": float(round(specificity, 4)),
            "f1_score": float(round(f1, 4))
        }

    except Exception as e:
        print("Metric error:", e)
        return {
            "accuracy": 0,
            "precision": 0,
            "recall": 0,
            "specificity": 0,
            "f1_score": 0
        }

# =========================
# FILE SERVERS
# =========================
@app.route("/uploads/<path:filename>")
def uploads(filename):
    resp = make_response(send_from_directory(UPLOAD_FOLDER, filename))
    resp.headers["Cache-Control"] = "no-store"
    return resp

@app.route("/outputs/<path:filename>")
def outputs(filename):
    resp = make_response(send_from_directory(OUTPUT_FOLDER, filename))
    resp.headers["Cache-Control"] = "no-store"
    return resp

# =========================
# HEALTH
# =========================
@app.route("/health")
def health():
    return jsonify({"status": "running"})

# =========================
# TREATMENT LOGIC
# =========================
def get_treatment(pred_class, confidence):

    if confidence < 0.70:
        return {
            "urgency": "High",
            "action": "Manual Review",
            "details": "Low confidence prediction"
        }

    return {
        "glioma": {
            "urgency": "High",
            "action": "Neuro-Oncology",
            "details": "Aggressive tumor suspected"
        },
        "meningioma": {
            "urgency": "Medium",
            "action": "Neurosurgery",
            "details": "Extra axial mass"
        },
        "pituitary": {
            "urgency": "Medium",
            "action": "Endocrine check",
            "details": "Pituitary anomaly"
        },
        "notumor": {
            "urgency": "Low",
            "action": "Routine check",
            "details": "No tumor detected"
        }
    }.get(pred_class, {
        "urgency": "Low",
        "action": "Unknown",
        "details": "N/A"
    })

# =========================
# PREDICT ENDPOINT
# =========================
@app.route("/predict", methods=["POST"])
def predict():

    try:
        if "file" not in request.files:
            return jsonify({"error": "No file provided"}), 400

        file = request.files["file"]
        if file.filename == "":
            return jsonify({"error": "Empty filename"}), 400

        ext = os.path.splitext(file.filename)[1]
        filename = f"{uuid.uuid4().hex}{ext}"
        file_path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(file_path)

        img = preprocess_image(file_path)
        img_array = np.expand_dims(img, axis=0)

        preds = model.predict(img_array, verbose=0)[0]

        idx = int(np.argmax(preds))
        predicted_class = CLASS_NAMES[idx]
        confidence = float(preds[idx])

        has_tumor = predicted_class != "notumor"

        prediction_text = (
            f"Positive: {predicted_class.upper()} DETECTED"
            if has_tumor else
            "Negative: NO TUMOR DETECTED"
        )

        # =====================
        # Grad-CAM
        # =====================
        heatmap_url = ""
        fallback = ""

        if has_tumor:
            try:
                heatmap_file = f"heatmap_{filename}"
                heatmap_path = os.path.join(OUTPUT_FOLDER, heatmap_file)

                heatmap = get_gradcam_heatmap(model, img_array, idx)
                save_and_overlay_heatmap(file_path, heatmap, heatmap_path)

                heatmap_url = f"/outputs/{heatmap_file}"

            except Exception as e:
                fallback = str(e)
        else:
            fallback = "No tumor detected — heatmap not required"

        # =====================
        # RESPONSE
        # =====================
        return jsonify({
            "prediction": prediction_text,
            "has_tumor": has_tumor,
            "class": predicted_class,
            "confidence": confidence,
            "original_image_url": f"/uploads/{filename}",
            "heatmap_image_url": heatmap_url,
            "heatmap_fallback_message": fallback,
            "treatment": get_treatment(predicted_class, confidence),
            "metrics": fetch_real_validation_metrics()
        })

    except Exception as e:
        print("ERROR:", e)
        return jsonify({"error": str(e)}), 500

# =========================
# RUN
# =========================
if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)