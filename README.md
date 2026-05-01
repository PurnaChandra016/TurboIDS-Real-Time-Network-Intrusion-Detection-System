# TurboIDS — Real-Time Network Intrusion Detection System

This repository implements a complete intrusion detection pipeline that trains a neural network on network traffic data and optimizes it across 7 inference runtimes — from raw PyTorch all the way to production Triton serving.

---

## 1. The Dataset: NSL-KDD

NSL-KDD is the standard benchmark dataset for network intrusion detection research, fixing the redundancy problems found in the older KDD Cup 1999 dataset.

**Why NSL-KDD?** It contains 148,517 labeled network connection records split into 125,973 training and 22,544 test samples, each described by 41 features covering connection properties, traffic statistics, and protocol behavior.

**The Classification Task:** Every connection is labeled as either **Normal** or **Attack** (binary). Attack types include Denial of Service (DoS), Probe/Scanning, Remote-to-Local (R2L), and User-to-Root (U2R) intrusions.

**Preprocessing:** Three categorical features — `protocol_type`, `service`, and `flag` — are label-encoded using a shared encoder fitted on the union of train and test sets to prevent unseen-category errors. All 41 features are then standardized using `StandardScaler`, saved as `scaler.pkl` for inference reuse.

---

## 2. The Model: MLP Classifier

Standard deep learning architectures are designed for image or sequence data. For tabular network traffic features, a well-tuned Multilayer Perceptron outperforms heavier architectures while staying fast enough for real-time inference.

**Architecture (defined in `train.py`):**

- **Input Layer:** 41 features
- **Hidden Layer 1:** 256 neurons → ReLU → Dropout
- **Hidden Layer 2:** 128 neurons → ReLU → Dropout
- **Hidden Layer 3:** 64 neurons → ReLU → Dropout
- **Output Layer:** 1 neuron (raw logit, sigmoid at inference)

**Class Imbalance Handling:** Instead of hardcoding a weight, `pos_weight` is computed dynamically at runtime as `num_normal / num_attack`. This makes the training script automatically adapt to any version of the dataset.

**Loss Function:** `BCEWithLogitsLoss` with dynamic `pos_weight`, trained using the Adam optimizer on GPU (CUDA).

---

## 3. Inference Optimization Pipeline

The trained model goes through a **7-stage optimization pipeline**, benchmarked with 1,000 timed runs and 100 warmup iterations at batch size 1. All results are written to a single unified `inference_results.json`.

**Stage 1 — PyTorch GPU:** Native PyTorch inference on GPU. Serves as the baseline for all speedup calculations.

**Stage 2 — ONNX CPU:** Model exported to ONNX format and executed on CPU using ONNX Runtime. Eliminates GPU transfer overhead for single-sample inference.

**Stage 3 — ONNX CUDA:** Same ONNX model run on GPU using the CUDA Execution Provider in ONNX Runtime.

**Stage 4 — TensorRT FP32:** ONNX model compiled into a TensorRT engine using full 32-bit precision. Engine is cached to disk — if the `.trt` file already exists, it is loaded directly without recompilation, saving several minutes on repeated runs.

**Stage 5 — TensorRT FP16:** Same TensorRT compilation with half-precision mode enabled, trading a small amount of numerical precision for speed.

**Stage 6 — Triton FP32:** TensorRT FP32 engine deployed on NVIDIA Triton Inference Server, adding production-grade HTTP/gRPC serving, health checks, and concurrent request handling.

**Stage 7 — Triton FP16:** Same as Stage 6 using the FP16 engine. Triton results are appended to the same `inference_results.json` produced by Stages 1–5, creating a single unified benchmark record.

---

## 4. Empirical Results

The pipeline was benchmarked with 1,000 timed runs and 100 warmup iterations at batch size 1:

```
========================= BENCHMARK REPORT =========================
Dataset: NSL-KDD | Batch Size: 1 | Runs: 1000
--------------------------------------------------------------------
Runtime         | Latency (ms) | Throughput (req/s) | Speedup
--------------------------------------------------------------------
PyTorch GPU     | 1.197        | 836                | 1.0x (baseline)
ONNX CPU        | 0.057        | 17,658             | 21.1x
ONNX CUDA       | 0.602        | 1,661              | 2.0x
TensorRT FP32   | 0.269        | 3,723              | 4.5x
TensorRT FP16   | 0.305        | 3,282              | 3.9x
Triton FP32     | 2.666        | 375                | 0.4x
Triton FP16     | 2.788        | 359                | 0.4x
====================================================================
```

**Key Findings:**

**ONNX CPU dominated single-sample latency** at 0.057ms with a 21.1x speedup. For a lightweight MLP on tabular data, CPU execution eliminates GPU transfer overhead entirely and outperforms all GPU runtimes at batch size 1.

**TensorRT FP32 delivered the best GPU-optimized latency** at 0.269ms (4.5x speedup). The gap between FP32 and FP16 was marginal, confirming that half-precision does not significantly hurt this model size.

**Triton showed higher per-request latency** due to HTTP/gRPC overhead, but its value is in production scalability — concurrent execution, dynamic batching, and multi-model management — which single-request benchmarks do not capture.

---

## 5. Batch Prediction on Full Test Set

The deployed pipeline was evaluated on all 22,544 NSL-KDD test samples through the Streamlit dashboard:

```
============= TEST SET EVALUATION =============
Total Samples   : 22,544
Normal Traffic  : 13,812  (61.3%)
Attack Traffic  :  8,732  (38.7%)
===============================================
```

Attack probability scores were concentrated near 0.0 and 1.0, indicating high model confidence with minimal ambiguous mid-range predictions.

---

## 6. Dashboard

A **FastAPI + Streamlit** web dashboard was built on top of the pipeline:

- **CSV Batch Predict** — upload NSL-KDD test CSV, get predictions with a pie chart, histogram of attack probabilities, and downloadable results
- **Benchmark Results** — live graphs of all 7 runtime latency, throughput, and speedup comparisons loaded from `inference_results.json`
- **About** — project overview and run instructions

| Service   | Port |
|-----------|------|
| FastAPI   | 8080 |
| Triton    | 8000 |
| Streamlit | 8501 |

---

## 7. Hardware Notes

**TRT Engine Build Time:** Compiling a TensorRT engine from scratch takes several minutes. The pipeline caches the `.trt` file to disk so subsequent runs skip recompilation entirely.

**Port Conflict:** Triton occupies port 8000 by default. FastAPI is deliberately run on port 8080 to avoid conflict.

**GPU Requirement:** Stages 1, 3, 4, and 5 require a CUDA-capable GPU. Stage 2 (ONNX CPU) can run independently, and Stages 6–7 require a live Triton server instance.

**TensorRT & PyCUDA:** These cannot be installed via a standard `pip install`. They require the [NVIDIA TensorRT SDK](https://developer.nvidia.com/tensorrt) installed on your system first, after which you can pip install the local wheel files.

---

## 8. How to Run

**Install dependencies:**
```bash
pip install -r requirements.txt
```

> **Note:** `tensorrt` and `pycuda` require the NVIDIA TensorRT SDK installed separately before pip installing.

**Run the full pipeline:**
```bash
# Train the model
python train.py

# Export to ONNX + benchmark 5 runtimes (Stages 1–5)
python export_optimize.py

# Benchmark on Triton — requires Triton server running on port 8000 (Stages 6–7)
python 3_triton_client.py

# Prepare test CSV for dashboard upload
python prep_csv.py

# Start FastAPI backend (port 8080)
uvicorn api:app --port 8080

# Start Streamlit dashboard (port 8501)
streamlit run app.py
```

---

## Project Structure

```
TurboIDS/
├── train.py                  # Preprocessing + MLP training
├── export_optimize.py        # ONNX export + 5-runtime benchmark
├── 3_triton_client.py        # Triton benchmark (Stages 6–7)
├── 4_graphs.py               # Result graphs
├── api.py                    # FastAPI backend
├── app.py                    # Streamlit dashboard
├── prep_csv.py               # CSV prep for dashboard upload
├── requirements.txt
├── models/
│   ├── mlp_ids_best.pth
│   ├── mlp_ids.onnx
│   ├── engine_fp32.trt
│   ├── engine_fp16.trt
│   └── scaler.pkl
├── triton_models/
│   ├── mlp_ids_fp32/
│   │   ├── config.pbtxt
│   │   └── 1/model.plan
│   └── mlp_ids_fp16/
│       ├── config.pbtxt
│       └── 1/model.plan
├── results/
│   ├── inference_results.json
│   ├── graph1_latency.png
│   ├── graph2_throughput.png
│   ├── graph3_speedup.png
│   └── graph4_precision.png
└── data/
    ├── KDDTrain+.txt
    ├── KDDTest+.txt
    └── test_upload.csv
**Institution:** SRM Institute of Science and Technology, Kattankulathur
**Guide:** Ms. Asis Marceline V
**Year:** 2026
