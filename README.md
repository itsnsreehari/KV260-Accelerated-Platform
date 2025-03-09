# KV260 Accelerated Platform

---

## Overview

This repository contains FPGA-accelerated AI models for the Kria KV260 board, featuring Transformer and GPT-2 implementations, as well as efficient YOLOv5 inference on video and image inputs.

- **Target Platform:** Xilinx Kria KV260  
- **AI Models:** YOLOv5, Transformer, GPT-2  
- **Acceleration Method:** Xilinx DPU IP  
- **Optimization Goal:** Improved inference speed and efficiency  
- **Deployment:** Edge AI on FPGA  

---

## Features

- FPGA-accelerated AI inference
- Real-time object detection with YOLOv5 (Nano and Large variants)
- Transformer architecture optimized for the DPU
- GPT-2 model inference with optimized matrix multiplication for DPU offload
- Detailed hardware and software setup guide for seamless deployment

---

## Hardware and Software Requirements

### Hardware
- Kria KV260 Vision AI Starter Kit
- SD card (minimum 32GB recommended)
- USB to UART cable for serial console access
- Ethernet connection for data transfer and debugging
- Camera module (for real-time inference)

### Software
- Vitis AI 3.0
- PetaLinux 2022.1
- Python 3.8+
- PyTorch (for Transformer/GPT-2 development)
- Docker (for YOLOv5 inference setup)

---

## Setup Guide

### 1. Prepare the SD Card
- Flash the PetaLinux 2022.1 image onto the SD card.

### 2. Install Vitis AI Runtime
```bash
sudo apt-get install vitis-ai-runtime
```

### 3. Load the XCLBIN File
- Copy your `.xclbin` file to `/usr/lib/dpu.xclbin` on the KV260 board.

### 4. Set Environment Variables
```bash
export XLNX_VART_FIRMWARE=/usr/lib/dpu.xclbin
source /etc/vart.conf
```

### 5. Verify Installation
```bash
dpu_fingerprint
```

---

## Model Implementations

### YOLOv5 Inference
- **Model Versions:** YOLOv5 Nano, YOLOv5 Large

**Command for Image Inference:**
```bash
python yolov5_inference.py --model yolov5n --image input.jpg
```

### Transformer with DPU Acceleration
- Implemented a small attention layer to accelerate attention weight computation.
- Successfully deployed a lightweight language model using DPU offload for matrix multiplications (matmul).

**Attention Layer Implementation Results:**
- Input Tensor Shape: `torch.Size([2, 5, 8])`
- Attention Output Shape: `torch.Size([2, 5, 8])`
- Attention Weights Shape: `torch.Size([2, 5, 5])`

### GPT-2 Deployment
- Optimized GPT-2 model for deployment on KV260 using quantization techniques and DPU acceleration.

**Command for GPT-2 Inference:**
```bash
python gpt2_inference.py --text "The future of AI is..."
```

---

## Performance Metrics

| Model Variant | Precision | FPS (KV260) | Inference Time |
|----------------|------------|---------------|----------------|
| YOLOv5 Nano      | INT8       | 59.2 FPS          | 0.1268 sec      |
| YOLOv5 Large     | INT8       | 17.6 FPS        | 7.6 FPS (video) |
| GPT-2            | Optimized  | DPU Accelerated  | -               |

---

## Conclusion
By deploying YOLOv5, Transformer, and GPT-2 models on the KV260 with DPU acceleration, we achieve real-time inference performance with improved efficiency and minimal accuracy loss. This project demonstrates the capabilities of FPGA-based AI inference for edge applications.

---

## License
This project is licensed under the MIT License. See the `LICENSE` file for more details.

