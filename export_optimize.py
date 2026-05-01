import torch
import torch.nn as nn
import numpy as np
import onnx
import onnxruntime as ort
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import time, json, os
import warnings
warnings.filterwarnings('ignore')

MODELS_DIR  = r'C:\Users\purna\seai_project\models'
RESULTS_DIR = r'C:\Users\purna\seai_project\results'
ONNX_PATH   = f'{MODELS_DIR}/mlp_ids.onnx'
INPUT_DIM   = 41
RUNS        = 1000
TRT_LOGGER  = trt.Logger(trt.Logger.ERROR)
os.makedirs(RESULTS_DIR, exist_ok=True)

class IntrusionMLP(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1)
        )  
    def forward(self, x):
        return self.net(x)

device = torch.device('cuda')
model  = IntrusionMLP(INPUT_DIM).to(device)
model.load_state_dict(torch.load(f'{MODELS_DIR}/mlp_ids_best.pth', weights_only=True))
model.eval()
print("✓ Model loaded successfully")

input_np = np.random.randn(1, INPUT_DIM).astype(np.float32)
dummy    = torch.tensor(input_np).to(device)

print("\n" + "="*50)
print("STAGE 1: PyTorch GPU Baseline Inference")
print("="*50)

for _ in range(100):
    with torch.no_grad():
        model(dummy)

torch.cuda.synchronize()
start = time.perf_counter()
for _ in range(RUNS):
    with torch.no_grad():
        model(dummy)
torch.cuda.synchronize()
end = time.perf_counter()

pytorch_latency    = (end - start) / RUNS * 1000
pytorch_throughput = RUNS / (end - start)
print(f"  Latency:    {pytorch_latency:.4f} ms")
print(f"  Throughput: {pytorch_throughput:.2f} req/sec")

print("\n" + "="*50)
print("STAGE 2: Exporting Model to ONNX")
print("="*50)

torch.onnx.export(
    model, dummy, ONNX_PATH,
    input_names=['input'], output_names=['output'],
    dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}},
    opset_version=11
)
onnx.checker.check_model(onnx.load(ONNX_PATH))
print(f"  ✓ ONNX model exported and verified: {ONNX_PATH}")

print("\n" + "="*50)
print("STAGE 3: ONNX Runtime - CPU Inference")
print("="*50)

sess_cpu = ort.InferenceSession(ONNX_PATH, providers=['CPUExecutionProvider'])
for _ in range(100):
    sess_cpu.run(None, {'input': input_np})
start = time.perf_counter()
for _ in range(RUNS):
    sess_cpu.run(None, {'input': input_np})
end = time.perf_counter()
onnx_cpu_latency    = (end - start) / RUNS * 1000
onnx_cpu_throughput = RUNS / (end - start)
print(f"  Latency:    {onnx_cpu_latency:.4f} ms")
print(f"  Throughput: {onnx_cpu_throughput:.2f} req/sec")

print("\n" + "="*50)
print("STAGE 4: ONNX Runtime - CUDA GPU Inference")
print("="*50)

sess_cuda = ort.InferenceSession(
    ONNX_PATH,
    providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
)
for _ in range(100):
    sess_cuda.run(None, {'input': input_np})
start = time.perf_counter()
for _ in range(RUNS):
    sess_cuda.run(None, {'input': input_np})
end = time.perf_counter()
onnx_gpu_latency    = (end - start) / RUNS * 1000
onnx_gpu_throughput = RUNS / (end - start)
print(f"  Latency:    {onnx_gpu_latency:.4f} ms")
print(f"  Throughput: {onnx_gpu_throughput:.2f} req/sec")

print("\n" + "="*50)
print("STAGE 5: TensorRT Optimized - FP32")
print("="*50)

def build_engine(onnx_path, fp16=False):
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser  = trt.OnnxParser(network, TRT_LOGGER)
    config  = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)
    if fp16:
        config.set_flag(trt.BuilderFlag.FP16)
    with open(onnx_path, 'rb') as f:
        parser.parse(f.read())
    profile = builder.create_optimization_profile()
    profile.set_shape('input', (1, INPUT_DIM), (32, INPUT_DIM), (128, INPUT_DIM))
    config.add_optimization_profile(profile)
    print(f"  Building {'FP16' if fp16 else 'FP32'} engine — please wait...")
    engine_bytes = builder.build_serialized_network(network, config)
    print(f"  ✓ Engine built successfully")
    return engine_bytes

def get_engine(onnx_path, engine_path, fp16=False):
    if os.path.exists(engine_path):
        print(f"  Loading cached engine: {engine_path}")
        with open(engine_path, 'rb') as f:
            return f.read()
    engine_bytes = build_engine(onnx_path, fp16=fp16)
    with open(engine_path, 'wb') as f:
        f.write(engine_bytes)
    return engine_bytes

def benchmark_trt(engine_bytes):
    runtime = trt.Runtime(TRT_LOGGER)
    engine  = runtime.deserialize_cuda_engine(engine_bytes)
    context = engine.create_execution_context()
    context.set_input_shape('input', (1, INPUT_DIM))
    d_input  = cuda.mem_alloc(input_np.nbytes)
    output   = np.empty((1, 1), dtype=np.float32)
    d_output = cuda.mem_alloc(output.nbytes)
    stream   = cuda.Stream()
    context.set_tensor_address('input',  int(d_input))
    context.set_tensor_address('output', int(d_output))
    for _ in range(100):
        cuda.memcpy_htod_async(d_input, input_np, stream)
        context.execute_async_v3(stream_handle=stream.handle)
        cuda.memcpy_dtoh_async(output, d_output, stream)
        stream.synchronize()
    start = time.perf_counter()
    for _ in range(RUNS):
        cuda.memcpy_htod_async(d_input, input_np, stream)
        context.execute_async_v3(stream_handle=stream.handle)
        cuda.memcpy_dtoh_async(output, d_output, stream)
        stream.synchronize()
    end = time.perf_counter()
    return (end - start) / RUNS * 1000, RUNS / (end - start)

engine_fp32 = get_engine(ONNX_PATH, f'{MODELS_DIR}/engine_fp32.trt', fp16=False)
trt_fp32_latency, trt_fp32_throughput = benchmark_trt(engine_fp32)
print(f"  Latency:    {trt_fp32_latency:.4f} ms")
print(f"  Throughput: {trt_fp32_throughput:.2f} req/sec")

print("\n" + "="*50)
print("STAGE 6: TensorRT Optimized - FP16")
print("="*50)

engine_fp16 = get_engine(ONNX_PATH, f'{MODELS_DIR}/engine_fp16.trt', fp16=True)
trt_fp16_latency, trt_fp16_throughput = benchmark_trt(engine_fp16)
print(f"  Latency:    {trt_fp16_latency:.4f} ms")
print(f"  Throughput: {trt_fp16_throughput:.2f} req/sec")

print("\n" + "="*60)
print("         INFERENCE PERFORMANCE SUMMARY")
print("="*60)
print(f"{'Stage':<28} {'Latency (ms)':<16} {'Throughput (req/s)'}")
print("-"*60)
print(f"{'1. PyTorch GPU Baseline':<28} {pytorch_latency:<16.4f} {pytorch_throughput:.2f}")
print(f"{'2. ONNX CPU':<28} {onnx_cpu_latency:<16.4f} {onnx_cpu_throughput:.2f}")
print(f"{'3. ONNX CUDA':<28} {onnx_gpu_latency:<16.4f} {onnx_gpu_throughput:.2f}")
print(f"{'4. TensorRT FP32':<28} {trt_fp32_latency:<16.4f} {trt_fp32_throughput:.2f}")
print(f"{'5. TensorRT FP16':<28} {trt_fp16_latency:<16.4f} {trt_fp16_throughput:.2f}")
print("="*60)
print(f"\nSpeedup over PyTorch baseline:")
print(f"  TensorRT FP32: {pytorch_latency/trt_fp32_latency:.1f}x faster")
print(f"  TensorRT FP16: {pytorch_latency/trt_fp16_latency:.1f}x faster")

results = {
    'pytorch_gpu':   {'latency_ms': pytorch_latency,    'throughput': pytorch_throughput},
    'onnx_cpu':      {'latency_ms': onnx_cpu_latency,   'throughput': onnx_cpu_throughput},
    'onnx_cuda':     {'latency_ms': onnx_gpu_latency,   'throughput': onnx_gpu_throughput},
    'tensorrt_fp32': {'latency_ms': trt_fp32_latency,   'throughput': trt_fp32_throughput},
    'tensorrt_fp16': {'latency_ms': trt_fp16_latency,   'throughput': trt_fp16_throughput},
}
with open(f'{RESULTS_DIR}/inference_results.json', 'w') as f:
    json.dump(results, f, indent=2)
print(f"\n✓ Results saved to {RESULTS_DIR}/inference_results.json")