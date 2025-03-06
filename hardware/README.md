# Hardware Acceleration for B4096 on Kria KV260

## Overview
This folder contains the necessary hardware files for deploying and running deep learning inference on the Xilinx Kria KV260 using the **B4096** DPU architecture. These files enable optimized execution of AI workloads by leveraging the FPGA fabric for acceleration.

## Contents

| File Name   | Description |
|------------|------------|
| **dpu_b4096.xclbin** | Compiled hardware binary for the DPU B4096 architecture |
| **pl.dtbo** | Device tree overlay for enabling the PL (Programmable Logic) on the KV260 |
| **shell.json** | JSON configuration file describing the shell and DPU setup |

## Hardware Details
- **Architecture:** B4096 (Optimized for high-performance AI inference on KV260)
- **FPGA Overlay:** Custom-built using Xilinx Vivado and Vitis AI
- **Compute Units:** 4 DPU cores for parallel execution
- **Memory Optimization:** Efficient DDR bandwidth utilization for high-throughput processing

## Deployment Instructions

### 1. Copy Hardware Files to KV260
Ensure that the hardware files are transferred to the target device:
```bash
scp dpu_b4096.xclbin pl.dtbo shell.json ubuntu@<kv260-ip>:~/hardware/
```

### 2. Load the FPGA Bitstream
On the KV260, run the following commands to load the bitstream and enable the DPU:
```bash
cd ~/hardware
sudo xmutil unloadapp
sudo xmutil loadapp dpu_b4096.xclbin
```

### 3. Apply the Device Tree Overlay
```bash
sudo dtbo-manager add pl.dtbo
```
Verify that the DPU is correctly loaded:
```bash
xbutil examine
```

## Validation
To ensure that the hardware is correctly configured, run a basic Vitis AI test:
```bash
python3 -c "import vart; print(vart.get_xclbin_path())"
```
If the hardware is loaded properly, the path to `dpu_b4096.xclbin` should be displayed.

## Performance Metrics
| Metric | Value |
|--------|--------|
| **DPU Clock** | 325 MHz |
| **Peak Throughput** | ~5 TOPS |
| **Latency (YOLOv5 nano)** | ~15 ms |

## Conclusion
This hardware setup provides an optimized inference environment for deploying AI models on the Kria KV260. The B4096 architecture ensures high performance with low latency, making it ideal for real-time applications.
