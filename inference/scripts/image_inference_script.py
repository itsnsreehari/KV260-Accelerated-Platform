import os
import math
import threading
import time
import sys
from ctypes import *
from typing import List
import cv2
import numpy as np
import xir
import vart

"""
Calculate sigmoid
"""
def sigmoid(x):
    return 1 / (1 + math.exp(-x))

"""
Get script directory
"""
def get_script_directory():
    path = os.getcwd()
    return path

"""
Pre-process input for YOLOv5
"""
def preprocess_yolov5_image(image_path, width=640, height=640):
    # Read the image
    image = cv2.imread(image_path)
    
    # Check if image is loaded
    if image is None:
        raise FileNotFoundError(f"Image not found or cannot be read: {image_path}")
    
    # Resize and normalize the image
    image_resized = cv2.resize(image, (width, height))
    image = image_resized.astype(np.float32) / 255.0  # Normalize to [0, 1]
    image = np.transpose(image, (2, 0, 1))  # Change data layout to CHW
    image = np.expand_dims(image, axis=0).astype(np.float32)  # Add batch dimension
    return image, image_resized

"""
Post-process YOLOv5 output to create bounding boxes
"""
def postprocess_yolov5_output(output, orig_image, input_width, input_height, conf_threshold=0.5, iou_threshold=0.4):
    orig_h, orig_w = orig_image.shape[:2]
    scale_w, scale_h = orig_w / input_width, orig_h / input_height

    bboxes = []
    for detection in output:
    
        print(f"detection shape: {detection.shape}")
          
        if len(detection) !=54 :
          continue
          
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]
        if confidence > conf_threshold:
            center_x, center_y, width, height = detection[:4]
            x_min = int((center_x - width / 2) * scale_w)
            y_min = int((center_y - height / 2) * scale_h)
            x_max = int((center_x + width / 2) * scale_w)
            y_max = int((center_y + height / 2) * scale_h)
            bboxes.append([x_min, y_min, x_max, y_max, confidence, class_id])

    return bboxes

"""
Draw bounding boxes on the image
"""
def draw_bounding_boxes(image, bboxes, classes):
    for bbox in bboxes:
        x_min, y_min, x_max, y_max, conf, class_id = bbox
        label = f"{classes[class_id]}: {conf:.2f}"
        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
        cv2.putText(image, label, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return image

"""
Run YOLOv5 with batch
"""
def run_yolov5(runner, img_list, cnt, classes):
    input_tensors = runner.get_input_tensors()
    output_tensors = runner.get_output_tensors()

    input_ndim = tuple(input_tensors[0].dims)
    output_ndim = tuple(output_tensors[0].dims)

    batch_size = input_ndim[0]
    output_size = output_ndim[1]

    n_of_images = len(img_list)
    count = 0
    total_time = 0  # To track total time

    while count < cnt:
        run_size = min(batch_size, n_of_images - count)
        input_data = [np.empty(input_ndim, dtype=np.float32, order="C")]
        output_data = [np.empty(output_ndim, dtype=np.float32, order="C")]

        for i in range(run_size):
            # Ensure the image shape is (batch_size, 3, 640, 640) for input tensor
            input_data[0][i, ...] = img_list[count + i][0].transpose((0,3,2,1))

        start_time = time.time()  # Record start time for inference
        job_id = runner.execute_async(input_data, output_data)
        runner.wait(job_id)
        inference_time = time.time() - start_time  # Calculate time for inference
        total_time += inference_time

        for i in range(run_size):
            output = output_data[0][i]
            orig_image = img_list[count + i][1]
            bboxes = postprocess_yolov5_output(output, orig_image, 640, 640)
            result_image = draw_bounding_boxes(orig_image, bboxes, classes)
            save_path = f"/home/ubuntu/result_{count + i}.jpeg"
            cv2.imwrite(save_path, result_image)

        count += run_size

    # Calculate FPS
    fps = cnt / total_time
    print(f"Total inference time: {total_time:.4f} seconds")
    print(f"FPS: {fps:.2f}")

"""
Obtain DPU subgraph
"""
def get_child_subgraph_dpu(graph: "Graph") -> List["Subgraph"]:
    assert graph is not None, "'graph' should not be None."
    root_subgraph = graph.get_root_subgraph()
    assert root_subgraph is not None, "Failed to get root subgraph of input Graph object."
    if root_subgraph.is_leaf:
        return []
    child_subgraphs = root_subgraph.toposort_child_subgraph()
    return [
        cs for cs in child_subgraphs
        if cs.has_attr("device") and cs.get_attr("device").upper() == "DPU"
    ]

"""
Main function
"""
def main(argv):
    global threadnum

    calib_image_dir = "/home/ubuntu/yolov5_scripts"   # Updated for specific image directory
    image_path = "/home/ubuntu/yolov5_scripts/test1.jpeg"  # Specific image path
    class_file = os.path.join(calib_image_dir, "words.txt")

    with open(class_file, "r") as f:
        classes = [line.strip() for line in f.readlines()]

    graph = xir.Graph.deserialize(argv[2])
    subgraphs = get_child_subgraph_dpu(graph)
    assert len(subgraphs) == 1, "Only one DPU kernel is supported."

    dpu_runner = vart.Runner.create_runner(subgraphs[0], "run")

    # Process single image
    try:
        img, orig_img = preprocess_yolov5_image(image_path)
        img_list = [(img, orig_img)]

        cnt = 1  # Single iteration for one image
        run_yolov5(dpu_runner, img_list, cnt, classes)
    except FileNotFoundError as e:
        print(e)
    except Exception as e:
        print(f"Error: {e}")

if _name_ == "_main_":
    if len(sys.argv) != 3:
        print("Usage: python3 yolov5.py <thread_number> <yolov5_xmodel_file>")
    else:
        main(sys.argv)
