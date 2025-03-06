import cv2
import numpy as np
import xir
import vart
import time
import sys

# Function to preprocess frame (pads to 640x640 while keeping aspect ratio)
def preprocess_frame(frame, input_height=640, input_width=640):
    orig_h, orig_w = frame.shape[:2]

    # Create a blank 640x640 black image
    padded_frame = np.zeros((input_height, input_width, 3), dtype=np.uint8)

    # Compute padding sizes (center the frame)
    top_pad = (input_height - orig_h) // 2
    bottom_pad = input_height - orig_h - top_pad
    left_pad = (input_width - orig_w) // 2
    right_pad = input_width - orig_w - left_pad

    # Place original frame in the center
    padded_frame[top_pad:top_pad + orig_h, left_pad:left_pad + orig_w] = frame

    # Normalize and convert to float32
    frame_normalized = padded_frame.astype(np.float32) / 255.0

    # Convert HWC to NHWC (because your model expects NHWC: (1, 640, 640, 3))
    frame_nhwc = np.expand_dims(frame_normalized, axis=0)  # Add batch dimension

    return frame_nhwc

# Function to process model output and extract bounding boxes
def postprocess_output(output, conf_threshold=0.5):
    bboxes = []
    for detection in output.reshape(-1, output.shape[-1]):  # Flatten for easier processing
        if len(detection) < 5:
            continue
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]

        if confidence > conf_threshold:
            center_x, center_y, width, height = detection[:4]
            x_min = int(center_x - width / 2)
            y_min = int(center_y - height / 2)
            x_max = int(center_x + width / 2)
            y_max = int(center_y + height / 2)
            bboxes.append([x_min, y_min, x_max, y_max, confidence, class_id])
    
    return bboxes

# Function to draw bounding boxes on frame
def draw_bounding_boxes(frame, bboxes, class_labels):
    for bbox in bboxes:
        x_min, y_min, x_max, y_max, conf, class_id = bbox
        class_name = class_labels[class_id] if class_id < len(class_labels) else f"Class {class_id}"
        label = f"{class_name}: {conf:.2f}"
        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
        cv2.putText(frame, label, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    return frame

# Function to run inference on a video using DPU
def run_inference(runner, video_path, output_path, class_labels):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video file: {video_path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    input_tensors = runner.get_input_tensors()
    output_tensors = runner.get_output_tensors()

    input_ndim = tuple(input_tensors[0].dims)
    output_ndim = tuple(output_tensors[0].dims)

    batch_size = input_ndim[0]
    input_height, input_width = input_ndim[1], input_ndim[2]

    print(f"Model Input Shape: {input_ndim}")
    print(f"Model Output Shape: {output_ndim}")

    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        if frame_count % 10 == 0:  # Only print every 10 frames to avoid spamming
            print(f"Processing frame {frame_count}: {frame.shape}")

        preprocessed_frame = preprocess_frame(frame, input_height, input_width)

        input_data = [np.empty(input_ndim, dtype=np.float32, order="C")]
        output_data = [np.empty(output_ndim, dtype=np.float32, order="C")]

        # Fixing shape alignment to NHWC
        input_data[0][...] = preprocessed_frame

        start_time = time.time()
        job_id = runner.execute_async(input_data, output_data)
        runner.wait(job_id)
        end_time = time.time()

        output = output_data[0]
        bboxes = postprocess_output(output)
        frame_with_boxes = draw_bounding_boxes(frame, bboxes, class_labels)

        out.write(frame_with_boxes)

    cap.release()
    out.release()
    print(f"Inference complete. Output saved to {output_path}")

# Function to get the DPU subgraph
def get_dpu_subgraph(graph):
    assert graph is not None, "'graph' should not be None."
    root_subgraph = graph.get_root_subgraph()
    assert root_subgraph is not None, "Failed to get root subgraph."
    if root_subgraph.is_leaf:
        return []
    child_subgraphs = root_subgraph.toposort_child_subgraph()
    return [cs for cs in child_subgraphs if cs.has_attr("device") and cs.get_attr("device").upper() == "DPU"]

# Main function
def main(argv):
    if len(argv) != 3:
        print("Usage: python3 code_for_video.py <thread_number> <yolov5_xmodel_file>")
        sys.exit(1)

    video_path = "/home/ubuntu/download.avi"
    output_path = "/home/ubuntu/outputvid.mp4"
    class_file = "/home/ubuntu/yolov5_scripts/words.txt"

    # Load class labels
    with open(class_file, "r") as f:
        class_labels = [line.strip() for line in f.readlines()]

    # Load YOLOv5 model
    graph = xir.Graph.deserialize(argv[2])
    subgraphs = get_dpu_subgraph(graph)
    assert len(subgraphs) == 1, "Only one DPU kernel is supported."

    # Create DPU runner
    dpu_runner = vart.Runner.create_runner(subgraphs[0], "run")

    try:
        run_inference(dpu_runner, video_path, output_path, class_labels)
    except FileNotFoundError as e:
        print(e)
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main(sys.argv)
