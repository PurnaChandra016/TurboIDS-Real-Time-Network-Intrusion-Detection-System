import streamlit as st
import requests
import json
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from io import StringIO
import time

st.set_page_config(
    page_title="TurboIDS Dashboard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

API_URL = "http://localhost:8080"

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Inter:wght@300;400;600&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

h1, h2, h3 {
    font-family: 'Space Mono', monospace;
}

.stApp {
    background-color: #0a0e1a;
    color: #e0e6f0;
}

.metric-card {
    background: linear-gradient(135deg, #0f1629 0%, #1a2340 100%);
    border: 1px solid #2a3a5c;
    border-radius: 12px;
    padding: 20px;
    text-align: center;
    margin: 5px 0;
}

.metric-card .value {
    font-family: 'Space Mono', monospace;
    font-size: 2rem;
    font-weight: 700;
    color: #4fc3f7;
}

.metric-card .label {
    font-size: 0.85rem;
    color: #7a8ba8;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-top: 6px;
}

.attack-badge {
    background: linear-gradient(135deg, #7b0000, #c62828);
    color: white;
    padding: 10px 24px;
    border-radius: 8px;
    font-family: 'Space Mono', monospace;
    font-size: 1.4rem;
    font-weight: 700;
    display: inline-block;
    letter-spacing: 2px;
}

.normal-badge {
    background: linear-gradient(135deg, #004d00, #2e7d32);
    color: white;
    padding: 10px 24px;
    border-radius: 8px;
    font-family: 'Space Mono', monospace;
    font-size: 1.4rem;
    font-weight: 700;
    display: inline-block;
    letter-spacing: 2px;
}

.stButton > button {
    background: linear-gradient(135deg, #1565c0, #0d47a1);
    color: white;
    border: none;
    border-radius: 8px;
    font-family: 'Space Mono', monospace;
    font-weight: 700;
    padding: 10px 28px;
    font-size: 0.9rem;
    letter-spacing: 1px;
    transition: all 0.2s;
}

.stButton > button:hover {
    background: linear-gradient(135deg, #1976d2, #1565c0);
    transform: translateY(-1px);
}

.sidebar-header {
    font-family: 'Space Mono', monospace;
    font-size: 1.1rem;
    color: #4fc3f7;
    border-bottom: 1px solid #2a3a5c;
    padding-bottom: 8px;
    margin-bottom: 12px;
}

div[data-testid="stSidebar"] {
    background-color: #080c18;
    border-right: 1px solid #1a2340;
}

.stNumberInput label, .stSelectbox label, .stFileUploader label {
    color: #a0b0c8 !important;
    font-size: 0.8rem;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}

.banner {
    background: linear-gradient(135deg, #0d1b3e 0%, #0a1628 50%, #0d1b3e 100%);
    border: 1px solid #1e3a6e;
    border-radius: 16px;
    padding: 28px 36px;
    margin-bottom: 28px;
    position: relative;
    overflow: hidden;
}

.banner::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
    background: linear-gradient(90deg, #1565c0, #4fc3f7, #1565c0);
}

.banner h1 {
    font-family: 'Space Mono', monospace;
    font-size: 2rem;
    color: #e8f0ff;
    margin: 0 0 6px 0;
    letter-spacing: 2px;
}

.banner p {
    color: #7a8ba8;
    margin: 0;
    font-size: 0.9rem;
}

.section-title {
    font-family: 'Space Mono', monospace;
    font-size: 1rem;
    color: #4fc3f7;
    text-transform: uppercase;
    letter-spacing: 2px;
    border-left: 3px solid #1565c0;
    padding-left: 12px;
    margin: 24px 0 16px 0;
}

.result-box {
    background: #0f1629;
    border: 1px solid #2a3a5c;
    border-radius: 12px;
    padding: 24px;
    text-align: center;
    margin-top: 16px;
}

.stDataFrame {
    border: 1px solid #2a3a5c;
    border-radius: 8px;
}
</style>
""", unsafe_allow_html=True)

FEATURES = [
    ("duration",                    "float", "Duration of connection (seconds)"),
    ("protocol_type",               "int",   "Protocol type: 0=icmp, 1=tcp, 2=udp"),
    ("service",                     "int",   "Network service (label-encoded 0–65)"),
    ("flag",                        "int",   "Connection status flag (label-encoded 0–10)"),
    ("src_bytes",                   "float", "Bytes from source to dest"),
    ("dst_bytes",                   "float", "Bytes from dest to source"),
    ("land",                        "int",   "1 if src/dest host+port same, else 0"),
    ("wrong_fragment",              "float", "Number of wrong fragments"),
    ("urgent",                      "float", "Number of urgent packets"),
    ("hot",                         "float", "Number of hot indicators"),
    ("num_failed_logins",           "float", "Number of failed login attempts"),
    ("logged_in",                   "int",   "1 if logged in, else 0"),
    ("num_compromised",             "float", "Number of compromised conditions"),
    ("root_shell",                  "int",   "1 if root shell obtained, else 0"),
    ("su_attempted",                "int",   "1 if su root command attempted, else 0"),
    ("num_root",                    "float", "Number of root accesses"),
    ("num_file_creations",          "float", "Number of file creation operations"),
    ("num_shells",                  "float", "Number of shell prompts"),
    ("num_access_files",            "float", "Number of operations on access control files"),
    ("num_outbound_cmds",           "float", "Number of outbound commands in ftp session"),
    ("is_host_login",               "int",   "1 if login is to host, else 0"),
    ("is_guest_login",              "int",   "1 if guest login, else 0"),
    ("count",                       "float", "Connections to same host (last 2 sec)"),
    ("srv_count",                   "float", "Connections to same service (last 2 sec)"),
    ("serror_rate",                 "float", "% connections with SYN errors (same host)"),
    ("srv_serror_rate",             "float", "% connections with SYN errors (same service)"),
    ("rerror_rate",                 "float", "% connections with REJ errors (same host)"),
    ("srv_rerror_rate",             "float", "% connections with REJ errors (same service)"),
    ("same_srv_rate",               "float", "% connections to same service (same host)"),
    ("diff_srv_rate",               "float", "% connections to diff services (same host)"),
    ("srv_diff_host_rate",          "float", "% connections to diff hosts (same service)"),
    ("dst_host_count",              "float", "Connections to same dest host (last 100)"),
    ("dst_host_srv_count",          "float", "Connections to same service on dest host"),
    ("dst_host_same_srv_rate",      "float", "% connections to same service on dest host"),
    ("dst_host_diff_srv_rate",      "float", "% connections to diff services on dest host"),
    ("dst_host_same_src_port_rate", "float", "% connections with same src port on dest host"),
    ("dst_host_srv_diff_host_rate", "float", "% connections to diff hosts on same service"),
    ("dst_host_serror_rate",        "float", "% connections with SYN errors on dest host"),
    ("dst_host_srv_serror_rate",    "float", "% SYN errors on same service on dest host"),
    ("dst_host_rerror_rate",        "float", "% REJ errors on dest host"),
    ("dst_host_srv_rerror_rate",    "float", "% REJ errors on same service on dest host"),
]

FEATURE_NAMES = [f[0] for f in FEATURES]

def check_api():
    try:
        r = requests.get(f"{API_URL}/health", timeout=2)
        return r.status_code == 200
    except:
        return False

def plot_benchmark(data):
    labels = {
        "pytorch_gpu":   "PyTorch GPU",
        "onnx_cpu":      "ONNX CPU",
        "onnx_cuda":     "ONNX CUDA",
        "tensorrt_fp32": "TensorRT FP32",
        "tensorrt_fp16": "TensorRT FP16",
        "triton_fp32":   "Triton FP32",
        "triton_fp16":   "Triton FP16",
    }
    colors = ["#4fc3f7","#81c784","#81c784","#ffb74d","#ffb74d","#ef5350","#ef5350"]
    keys = [k for k in labels if k in data]
    names = [labels[k] for k in keys]
    latencies = [data[k]["latency_ms"] for k in keys]
    throughputs = [data[k]["throughput"] for k in keys]

    fig_lat = go.Figure(go.Bar(
        x=names, y=latencies, marker_color=colors[:len(keys)],
        text=[f"{v:.3f}ms" for v in latencies], textposition="outside"
    ))
    fig_lat.update_layout(
        title="Inference Latency (lower is better)",
        paper_bgcolor="#0a0e1a", plot_bgcolor="#0f1629",
        font=dict(color="#a0b0c8", family="Space Mono"),
        yaxis=dict(title="Latency (ms)", gridcolor="#1a2340"),
        xaxis=dict(gridcolor="#1a2340"),
        margin=dict(t=50, b=10)
    )

    fig_thr = go.Figure(go.Bar(
        x=names, y=throughputs, marker_color=colors[:len(keys)],
        text=[f"{int(v)}" for v in throughputs], textposition="outside"
    ))
    fig_thr.update_layout(
        title="Throughput (higher is better)",
        paper_bgcolor="#0a0e1a", plot_bgcolor="#0f1629",
        font=dict(color="#a0b0c8", family="Space Mono"),
        yaxis=dict(title="Requests/sec", gridcolor="#1a2340"),
        xaxis=dict(gridcolor="#1a2340"),
        margin=dict(t=50, b=10)
    )

    baseline = data.get("pytorch_gpu", {}).get("latency_ms", 1)
    speedups = [baseline / data[k]["latency_ms"] for k in keys]
    fig_spd = go.Figure(go.Bar(
        x=names, y=speedups, marker_color=colors[:len(keys)],
        text=[f"{v:.1f}x" for v in speedups], textposition="outside"
    ))
    fig_spd.add_hline(y=1.0, line_dash="dash", line_color="#ffffff", opacity=0.4)
    fig_spd.update_layout(
        title="Speedup vs PyTorch Baseline",
        paper_bgcolor="#0a0e1a", plot_bgcolor="#0f1629",
        font=dict(color="#a0b0c8", family="Space Mono"),
        yaxis=dict(title="Speedup (x)", gridcolor="#1a2340"),
        xaxis=dict(gridcolor="#1a2340"),
        margin=dict(t=50, b=10)
    )

    return fig_lat, fig_thr, fig_spd

with st.sidebar:
    st.markdown('<div class="sidebar-header">🛡️ TurboIDS</div>', unsafe_allow_html=True)

    api_ok = check_api()
    if api_ok:
        st.success("API Connected", icon="✅")
    else:
        st.error("API Offline — run: uvicorn api:app --port 8000", icon="❌")

    st.markdown("---")
    st.markdown('<div class="sidebar-header">Navigation</div>', unsafe_allow_html=True)
    page = st.radio("", ["CSV Batch Predict", "Benchmark Results", "About"], label_visibility="collapsed")

    st.markdown("---")
    st.markdown('<div style="color:#4a5a78; font-size:0.75rem; font-family:Space Mono">TurboIDS v1.0<br>NSL-KDD · MLP · ONNX<br>SEAI Project · 2026</div>', unsafe_allow_html=True)

st.markdown("""
<div class="banner">
  <h1>🛡️ TURBO<span style="color:#4fc3f7">IDS</span></h1>
  <p>Neural Network Intrusion Detection · NSL-KDD · Multi-Runtime Inference Optimization</p>
</div>
""", unsafe_allow_html=True)

if page == "CSV Batch Predict":
    st.markdown('<div class="section-title">Batch Prediction via CSV</div>', unsafe_allow_html=True)
    st.caption("Upload a CSV with 41 feature columns (no label column). Column names must match NSL-KDD feature names.")

    with st.expander("Expected column names"):
        st.code(", ".join(FEATURE_NAMES))

    uploaded = st.file_uploader("Upload CSV", type=["csv", "txt"])

    if uploaded:
        try:
            df = pd.read_csv(uploaded)
            st.success(f"Loaded {len(df)} rows, {len(df.columns)} columns")
            st.dataframe(df.head(5), use_container_width=True)

            available = [c for c in FEATURE_NAMES if c in df.columns]
            missing = [c for c in FEATURE_NAMES if c not in df.columns]

            if missing:
                st.warning(f"Missing columns: {missing}. They will be filled with 0.")
                for m in missing:
                    df[m] = 0

            if st.button("🔍 RUN BATCH PREDICTION", use_container_width=True):
                if not api_ok:
                    st.error("API is not running.")
                else:
                    rows = df[FEATURE_NAMES].fillna(0).values.tolist()
                    with st.spinner(f"Predicting {len(rows)} rows..."):
                        r = requests.post(f"{API_URL}/predict/batch", json={"rows": rows}, timeout=60)
                        results = r.json()["predictions"]

                    result_df = pd.DataFrame(results)
                    df_out = df.copy()
                    df_out["predicted_label"] = result_df["label"].values
                    df_out["attack_probability"] = result_df["attack_probability"].values

                    st.markdown('<div class="section-title">Results</div>', unsafe_allow_html=True)

                    attacks = int((result_df["label"] == "Attack").sum())
                    normals = int((result_df["label"] == "Normal").sum())

                    c1, c2, c3 = st.columns(3)
                    with c1:
                        st.markdown(f'<div class="metric-card"><div class="value">{len(rows)}</div><div class="label">Total Samples</div></div>', unsafe_allow_html=True)
                    with c2:
                        st.markdown(f'<div class="metric-card"><div class="value" style="color:#ef5350">{attacks}</div><div class="label">Attacks Detected</div></div>', unsafe_allow_html=True)
                    with c3:
                        st.markdown(f'<div class="metric-card"><div class="value" style="color:#4caf50">{normals}</div><div class="label">Normal Traffic</div></div>', unsafe_allow_html=True)

                    fig_pie = go.Figure(go.Pie(
                        labels=["Normal", "Attack"],
                        values=[normals, attacks],
                        marker_colors=["#4caf50", "#ef5350"],
                        hole=0.45
                    ))
                    fig_pie.update_layout(
                        paper_bgcolor="#0a0e1a",
                        font=dict(color="#a0b0c8", family="Space Mono"),
                        legend=dict(bgcolor="#0f1629"),
                        margin=dict(t=30, b=10),
                        height=300,
                        title="Classification Distribution"
                    )
                    st.plotly_chart(fig_pie, use_container_width=True)

                    fig_hist = go.Figure(go.Histogram(
                        x=result_df["attack_probability"],
                        nbinsx=30,
                        marker_color="#4fc3f7",
                        opacity=0.8
                    ))
                    fig_hist.add_vline(x=0.5, line_dash="dash", line_color="#ffeb3b", opacity=0.7)
                    fig_hist.update_layout(
                        title="Attack Probability Distribution",
                        paper_bgcolor="#0a0e1a", plot_bgcolor="#0f1629",
                        font=dict(color="#a0b0c8", family="Space Mono"),
                        xaxis=dict(title="Attack Probability", gridcolor="#1a2340"),
                        yaxis=dict(title="Count", gridcolor="#1a2340"),
                        margin=dict(t=50, b=10),
                        height=300
                    )
                    st.plotly_chart(fig_hist, use_container_width=True)

                    st.markdown('<div class="section-title">Full Results Table</div>', unsafe_allow_html=True)
                    st.dataframe(
                        df_out[["predicted_label", "attack_probability"] + FEATURE_NAMES[:5]],
                        use_container_width=True,
                        height=400
                    )

                    csv_out = df_out.to_csv(index=False)
                    st.download_button(
                        "⬇️ Download Results CSV",
                        data=csv_out,
                        file_name="turboids_predictions.csv",
                        mime="text/csv",
                        use_container_width=True
                    )

        except Exception as e:
            st.error(f"Error processing file: {e}")

elif page == "Benchmark Results": 
    st.markdown('<div class="section-title">Inference Benchmark Results</div>', unsafe_allow_html=True)
    st.caption("Performance across all 7 inference stages. Measured with 1,000 runs + 100 warmup at batch size 1.")

    data = None
    if api_ok:
        try:
            r = requests.get(f"{API_URL}/benchmark", timeout=5)
            if r.status_code == 200:
                data = r.json()
        except:
            pass

    if data is None:
        import os
        local_path = os.path.join(os.path.dirname(__file__), "results", "inference_results.json")
        if os.path.exists(local_path):
            with open(local_path) as f:
                data = json.load(f)
        else:
            st.warning("Could not load benchmark data. Make sure API is running or results/inference_results.json exists.")
            data = None

    if data:
        labels_map = {
            "pytorch_gpu":   "PyTorch GPU",
            "onnx_cpu":      "ONNX CPU",
            "onnx_cuda":     "ONNX CUDA",
            "tensorrt_fp32": "TensorRT FP32",
            "tensorrt_fp16": "TensorRT FP16",
            "triton_fp32":   "Triton FP32",
            "triton_fp16":   "Triton FP16",
        }
        keys = [k for k in labels_map if k in data]

        best_lat_key = min(keys, key=lambda k: data[k]["latency_ms"])
        best_thr_key = max(keys, key=lambda k: data[k]["throughput"])
        baseline = data.get("pytorch_gpu", {}).get("latency_ms", 1)
        best_speedup = max(baseline / data[k]["latency_ms"] for k in keys)

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.markdown(f'<div class="metric-card"><div class="value">{len(keys)}</div><div class="label">Runtimes Tested</div></div>', unsafe_allow_html=True)
        with c2:
            st.markdown(f'<div class="metric-card"><div class="value">{data[best_lat_key]["latency_ms"]:.3f}ms</div><div class="label">Best Latency<br><span style="font-size:0.7rem;color:#4fc3f7">{labels_map[best_lat_key]}</span></div></div>', unsafe_allow_html=True)
        with c3:
            st.markdown(f'<div class="metric-card"><div class="value">{int(data[best_thr_key]["throughput"]):,}</div><div class="label">Best Throughput<br><span style="font-size:0.7rem;color:#4fc3f7">{labels_map[best_thr_key]}</span></div></div>', unsafe_allow_html=True)
        with c4:
            st.markdown(f'<div class="metric-card"><div class="value">{best_speedup:.1f}x</div><div class="label">Max Speedup</div></div>', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        fig_lat, fig_thr, fig_spd = plot_benchmark(data)

        st.plotly_chart(fig_lat, use_container_width=True)
        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(fig_thr, use_container_width=True)
        with col2:
            st.plotly_chart(fig_spd, use_container_width=True)

        st.markdown('<div class="section-title">Raw Benchmark Data</div>', unsafe_allow_html=True)
        rows = []
        for k in keys:
            spd = baseline / data[k]["latency_ms"]
            rows.append({
                "Runtime": labels_map[k],
                "Latency (ms)": round(data[k]["latency_ms"], 4),
                "Throughput (req/s)": int(data[k]["throughput"]),
                "Speedup vs PyTorch": f"{spd:.1f}x"
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

elif page == "About":
    st.markdown('<div class="section-title">About TurboIDS</div>', unsafe_allow_html=True)

    st.markdown("""
    **TurboIDS** is a complete neural network intrusion detection pipeline built on the NSL-KDD dataset,
    optimized for production inference using ONNX, TensorRT, and NVIDIA Triton.

    ### Model
    - Architecture: MLP (41 → 256 → 128 → 64 → 1)
    - Dataset: NSL-KDD (125,973 train / 22,544 test)
    - Accuracy: **79.90%** | Attack Precision: **0.97**
    - Loss: BCEWithLogitsLoss with class weighting

    ### Inference Pipeline
    | Stage | Runtime | Notes |
    |-------|---------|-------|
    | 1 | PyTorch GPU | Baseline |
    | 2 | ONNX CPU | Best single-sample latency |
    | 3 | ONNX CUDA | GPU via ONNX Runtime |
    | 4 | TensorRT FP32 | ~4.9x speedup |
    | 5 | TensorRT FP16 | Comparable to FP32 |
    | 6 | Triton FP32 | Production serving |
    | 7 | Triton FP16 | Production serving |

    ### Project Structure
    ```
    seai/
    ├── train.py              # Training pipeline
    ├── export_optimize.py    # ONNX + TRT benchmark
    ├── 3_triton_client.py    # Triton benchmark
    ├── 4_graphs.py           # Result visualization
    ├── api.py                # FastAPI backend  ← NEW
    ├── app.py                # Streamlit frontend ← NEW
    ├── models/
    ├── results/
    └── data/
    ```

    ### How to Run
    ```bash
    pip install fastapi uvicorn streamlit plotly requests
    # Terminal 1
    uvicorn api:app --port 8000
    # Terminal 2
    streamlit run app.py
    ```

    """)