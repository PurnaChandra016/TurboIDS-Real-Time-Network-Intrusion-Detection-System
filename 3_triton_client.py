import numpy as np
import tritonclient.http as httpclient
import time
import json
import os

RESULTS_DIR = r'C:\Users\purna\seai_project\results'
INPUT_DIM = 41
RUNS = 1000

client = httpclient.InferenceServerClient(url="localhost:8000")
print("✓ Connected to Triton Inference Server")
print(f"  Server ready: {client.is_server_ready()}")
print(f"  mlp_ids_fp32 ready: {client.is_model_ready('mlp_ids_fp32')}")
print(f"  mlp_ids_fp16 ready: {client.is_model_ready('mlp_ids_fp16')}")

input_data = np.random.randn(1, INPUT_DIM).astype(np.float32)

def benchmark_triton(model_name, label):
    print(f"\n--- Benchmarking {label} ---")
    inputs = [httpclient.InferInput('input', [1, INPUT_DIM], 'FP32')]
    inputs[0].set_data_from_numpy(input_data)
    outputs = [httpclient.InferRequestedOutput('output')]

    for _ in range(100):
        client.infer(model_name, inputs, outputs=outputs)

    start = time.perf_counter()
    for _ in range(RUNS):
        client.infer(model_name, inputs, outputs=outputs)
    end = time.perf_counter()

    latency = (end - start) / RUNS * 1000
    throughput = RUNS / (end - start)
    print(f"  Latency:    {latency:.4f} ms")
    print(f"  Throughput: {throughput:.2f} req/sec")
    return latency, throughput

triton_fp32_lat, triton_fp32_thr = benchmark_triton('mlp_ids_fp32', 'Triton FP32')
triton_fp16_lat, triton_fp16_thr = benchmark_triton('mlp_ids_fp16', 'Triton FP16')

print("\n" + "="*50)
print("  TRITON BENCHMARK RESULTS")
print("="*50)
print(f"  Triton FP32: {triton_fp32_lat:.4f} ms | {triton_fp32_thr:.2f} req/s")
print(f"  Triton FP16: {triton_fp16_lat:.4f} ms | {triton_fp16_thr:.2f} req/s")

with open(f'{RESULTS_DIR}/inference_results.json', 'r') as f:
    results = json.load(f)

results['triton_fp32'] = {'latency_ms': triton_fp32_lat, 'throughput': triton_fp32_thr}
results['triton_fp16'] = {'latency_ms': triton_fp16_lat, 'throughput': triton_fp16_thr}

with open(f'{RESULTS_DIR}/inference_results.json', 'w') as f:
    json.dump(results, f, indent=2)
print(f"\n✓ Triton results saved to inference_results.json")