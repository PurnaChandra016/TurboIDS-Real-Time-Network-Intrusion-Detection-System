import json
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import os

RESULTS_DIR = r'C:\Users\purna\seai_project\results'

with open(f'{RESULTS_DIR}/inference_results.json', 'r') as f:
    data = json.load(f)

labels = ['PyTorch\nGPU', 'ONNX\nCPU', 'ONNX\nCUDA', 'TensorRT\nFP32', 'TensorRT\nFP16', 'Triton\nFP32', 'Triton\nFP16']
keys   = ['pytorch_gpu', 'onnx_cpu', 'onnx_cuda', 'tensorrt_fp32', 'tensorrt_fp16', 'triton_fp32', 'triton_fp16']

latencies    = [data[k]['latency_ms']  for k in keys]
throughputs  = [data[k]['throughput']  for k in keys]

colors = ['#4C72B0', '#55A868', '#55A868', '#DD8452', '#DD8452', '#C44E52', '#C44E52']

fig, ax = plt.subplots(figsize=(12, 6))
bars = ax.bar(labels, latencies, color=colors, edgecolor='white', linewidth=0.5)
ax.set_title('Inference Latency Comparison\n(lower is better)', fontsize=14, fontweight='bold', pad=15)
ax.set_ylabel('Latency (ms)', fontsize=12)
ax.set_xlabel('Inference Method', fontsize=12)
for bar, val in zip(bars, latencies):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
            f'{val:.3f}ms', ha='center', va='bottom', fontsize=9, fontweight='bold')

legend_patches = [
    mpatches.Patch(color='#4C72B0', label='PyTorch Baseline'),
    mpatches.Patch(color='#55A868', label='ONNX Runtime'),
    mpatches.Patch(color='#DD8452', label='TensorRT'),
    mpatches.Patch(color='#C44E52', label='Triton Server'),
]
ax.legend(handles=legend_patches, loc='upper right')
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(f'{RESULTS_DIR}/graph1_latency.png', dpi=150, bbox_inches='tight')
plt.close()
print("✓ Graph 1 saved: graph1_latency.png")

fig, ax = plt.subplots(figsize=(12, 6))
bars = ax.bar(labels, throughputs, color=colors, edgecolor='white', linewidth=0.5)
ax.set_title('Inference Throughput Comparison\n(higher is better)', fontsize=14, fontweight='bold', pad=15)
ax.set_ylabel('Throughput (requests/sec)', fontsize=12)
ax.set_xlabel('Inference Method', fontsize=12)
for bar, val in zip(bars, throughputs):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 100,
            f'{val:.0f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
ax.legend(handles=legend_patches, loc='upper right')
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(f'{RESULTS_DIR}/graph2_throughput.png', dpi=150, bbox_inches='tight')
plt.close()
print("✓ Graph 2 saved: graph2_throughput.png")

baseline_lat = data['pytorch_gpu']['latency_ms']
speedups = [baseline_lat / data[k]['latency_ms'] for k in keys]

fig, ax = plt.subplots(figsize=(12, 6))
bars = ax.bar(labels, speedups, color=colors, edgecolor='white', linewidth=0.5)
ax.axhline(y=1.0, color='black', linestyle='--', linewidth=1.5, label='PyTorch Baseline (1x)')
ax.set_title('Speedup over PyTorch GPU Baseline\n(higher is better)', fontsize=14, fontweight='bold', pad=15)
ax.set_ylabel('Speedup (x times faster)', fontsize=12)
ax.set_xlabel('Inference Method', fontsize=12)
for bar, val in zip(bars, speedups):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
            f'{val:.1f}x', ha='center', va='bottom', fontsize=10, fontweight='bold')
ax.legend(handles=legend_patches + [mpatches.Patch(color='black', label='Baseline (1x)')], loc='upper right')
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(f'{RESULTS_DIR}/graph3_speedup.png', dpi=150, bbox_inches='tight')
plt.close()
print("✓ Graph 3 saved: graph3_speedup.png")

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

precision_labels = ['TensorRT FP32', 'TensorRT FP16', 'Triton FP32', 'Triton FP16']
precision_keys   = ['tensorrt_fp32', 'tensorrt_fp16', 'triton_fp32', 'triton_fp16']
prec_latencies   = [data[k]['latency_ms']  for k in precision_keys]
prec_throughputs = [data[k]['throughput']  for k in precision_keys]
prec_colors      = ['#DD8452', '#E8A87C', '#C44E52', '#D97070']

bars1 = axes[0].bar(precision_labels, prec_latencies, color=prec_colors, edgecolor='white')
axes[0].set_title('FP32 vs FP16 Latency', fontsize=12, fontweight='bold')
axes[0].set_ylabel('Latency (ms)')
for bar, val in zip(bars1, prec_latencies):
    axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                 f'{val:.3f}ms', ha='center', va='bottom', fontsize=9, fontweight='bold')
axes[0].grid(axis='y', alpha=0.3)
axes[0].tick_params(axis='x', labelsize=9)

bars2 = axes[1].bar(precision_labels, prec_throughputs, color=prec_colors, edgecolor='white')
axes[1].set_title('FP32 vs FP16 Throughput', fontsize=12, fontweight='bold')
axes[1].set_ylabel('Throughput (req/s)')
for bar, val in zip(bars2, prec_throughputs):
    axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10,
                 f'{val:.0f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
axes[1].grid(axis='y', alpha=0.3)
axes[1].tick_params(axis='x', labelsize=9)

plt.suptitle('FP32 vs FP16 Precision Comparison', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{RESULTS_DIR}/graph4_precision.png', dpi=150, bbox_inches='tight')
plt.close()
print("✓ Graph 4 saved: graph4_precision.png")

print(f"\n✓ All 4 graphs saved to {RESULTS_DIR}")