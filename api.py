from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
import pickle
import onnxruntime as ort
import json
import os
import time
from typing import List

app = FastAPI(title="TurboIDS API", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
SCALER_PATH = os.path.join(BASE_DIR, "models", "scaler.pkl")
ONNX_PATH   = os.path.join(BASE_DIR, "models", "mlp_ids.onnx")
RESULTS_PATH= os.path.join(BASE_DIR, "results", "inference_results.json")

scaler = None
sess   = None

@app.on_event("startup")
def load_model():
    global scaler, sess
    with open(SCALER_PATH, "rb") as f:
        scaler = pickle.load(f)
    sess = ort.InferenceSession(ONNX_PATH, providers=["CPUExecutionProvider"])
    print("✓ Model and scaler loaded")

COLUMNS = [
    'duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes',
    'land', 'wrong_fragment', 'urgent', 'hot', 'num_failed_logins', 'logged_in',
    'num_compromised', 'root_shell', 'su_attempted', 'num_root', 'num_file_creations',
    'num_shells', 'num_access_files', 'num_outbound_cmds', 'is_host_login',
    'is_guest_login', 'count', 'srv_count', 'serror_rate', 'srv_serror_rate',
    'rerror_rate', 'srv_rerror_rate', 'same_srv_rate', 'diff_srv_rate',
    'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count',
    'dst_host_same_srv_rate', 'dst_host_diff_srv_rate',
    'dst_host_same_src_port_rate', 'dst_host_srv_diff_host_rate',
    'dst_host_serror_rate', 'dst_host_srv_serror_rate',
    'dst_host_rerror_rate', 'dst_host_srv_rerror_rate'
]

class PredictRequest(BaseModel):
    features: List[float]  # exactly 41 values

class BatchPredictRequest(BaseModel):
    rows: List[List[float]]  # list of 41-feature rows

@app.post("/predict")
def predict(req: PredictRequest):
    if len(req.features) != 41:
        raise HTTPException(status_code=400, detail=f"Expected 41 features, got {len(req.features)}")
    
    start = time.perf_counter()
    x = np.array([req.features], dtype=np.float32)
    x_scaled = scaler.transform(x).astype(np.float32)
    logit = sess.run(None, {"input": x_scaled})[0][0][0]
    latency_ms = (time.perf_counter() - start) * 1000

    prob = float(1 / (1 + np.exp(-logit)))
    label = "Attack" if prob > 0.5 else "Normal"

    return {
        "label": label,
        "confidence": round(prob if label == "Attack" else 1 - prob, 4),
        "attack_probability": round(prob, 4),
        "latency_ms": round(latency_ms, 4)
    }

@app.post("/predict/batch")
def predict_batch(req: BatchPredictRequest):
    results = []
    for i, row in enumerate(req.rows):
        if len(row) != 41:
            raise HTTPException(status_code=400, detail=f"Row {i}: expected 41 features, got {len(row)}")
        x = np.array([row], dtype=np.float32)
        x_scaled = scaler.transform(x).astype(np.float32)
        logit = sess.run(None, {"input": x_scaled})[0][0][0]
        prob = float(1 / (1 + np.exp(-logit)))
        label = "Attack" if prob > 0.5 else "Normal"
        results.append({
            "row": i,
            "label": label,
            "attack_probability": round(prob, 4)
        })
    return {"predictions": results, "total": len(results)}

@app.get("/benchmark")
def get_benchmark():
    if not os.path.exists(RESULTS_PATH):
        raise HTTPException(status_code=404, detail="inference_results.json not found")
    with open(RESULTS_PATH) as f:
        return json.load(f)

@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": sess is not None}

@app.get("/features")
def get_features():
    return {"features": COLUMNS}
