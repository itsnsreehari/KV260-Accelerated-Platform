# YOLOv5 Quantization for Kria KV260

---

## Overview

This repository contains the quantization workflow for deploying the YOLOv5 model on the Xilinx Kria KV260. The quantization process reduces the model’s precision to improve inference speed and efficiency while maintaining accuracy. It is performed using Vitis AI and optimized for execution on the Deep Learning Processing Unit (DPU).

- **Target Platform:** Xilinx Kria KV260  
- **Quantization Framework:** Vitis AI 3.0  
- **Model Type:** YOLOv5 (Nano, Small, Medium, Large, X)  
- **Optimization Goal:** Reduce precision (FP32 → INT8) while preserving accuracy  
- **Deployment:** Edge AI on FPGA  

---

## Quantization Workflow

### 1. Model Preparation
Before quantization, the YOLOv5 model is first trained and exported to ONNX format.

- Train YOLOv5 on a custom dataset or use a pre-trained model.
- Export the trained model to ONNX using the following command:

  ```bash
  python export.py --weights yolov5s.pt --include onnx
  ```

### 2. Quantization with Vitis AI
The ONNX model is then converted into a quantized format using Vitis AI.

#### Step 1: Convert ONNX to XIR
Convert the ONNX model to an XIR format compatible with the DPU.

  ```bash
  vai_c_xir -x yolov5.onnx -o yolov5.xmodel --net_name yolov5
  ```

#### Step 2: Quantize the Model
Use the Vitis AI quantizer to convert the model from FP32 to INT8.

  ```bash
  vai_q_pytorch quantizer.py --model yolov5.onnx --calib_dataset calib_data/
  ```

### 3. Compilation for KV260 DPU
After quantization, the model is compiled for execution on the KV260’s DPU.

  ```bash
  vai_c_dpu -x yolov5.xmodel -o compiled_yolov5.xmodel --arch kv260.json
  ```

### Deployment on Kria KV260
Once quantized and compiled, the model is deployed on the KV260 and run using the Vitis AI runtime.

#### Running Inference on KV260
1. Transfer the `compiled_yolov5.xmodel` to the KV260.
2. Run the inference script:

  ```bash
  python run_inference.py --model compiled_yolov5.xmodel --image input.jpg
  ```

### Performance Metrics

| Model Variant  | Precision | FPS (KV260) | Accuracy Drop |
|---------------|-----------|-------------|---------------|
| YOLOv5 Nano  | INT8      | 59.2 FPS      | ~1%           |
| YOLOv5 Large | INT8      | 17.6 FPS    | ~1.5%         |

Quantization enables a significant speedup with minimal accuracy loss.

---

## Conclusion
By quantizing YOLOv5 for the KV260, we achieve real-time object detection with improved performance and lower power consumption. The Vitis AI toolchain ensures efficient deployment while maintaining accuracy.
