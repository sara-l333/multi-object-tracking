import os
import cv2
import numpy as np
from trtutils.impls.yolo import YOLOX
import argparse
import time
import psutil

# initialize YOLOX model with CUDA
yolo = YOLOX("yolox.engine", preprocessor="cuda")
process = psutil.Process(os.getpid())

# set resolution and input dir from command line
parser = argparse.ArgumentParser()
parser.add_argument('--width', type=int, default=640, help='Resize width')
parser.add_argument('--height', type=int, default=480, help='Resize height')
parser.add_argument('--input_dir', type=str, required=True, help='Directory with input images')
parser.add_argument('--csv_log', type=str, default="frame_times.csv", help='Optional CSV log file')
args = parser.parse_args()
dataset_path = args.input_dir
target_size = (args.width, args.height)
csv_log = args.csv_log

output_dir = "yolox_detections/MOT20-05"
os.makedirs(output_dir, exist_ok=True)

# initialize CSV file
with open(csv_log, "w") as f:
    f.write("frame_id,width,height,load_time,resize_time,preprocess_time,infer_time,postproc_time,total_time,memory_MB\n")

# iterate over images
for frame_id, image_name in enumerate(sorted(os.listdir(dataset_path))):
    if image_name.endswith(".jpg"):
        frame_start = time.perf_counter()

        # load image
        image_path = os.path.join(dataset_path, image_name)
        img = cv2.imread(image_path)
        if img is None:
            print(f"Failed to load image {image_name}")
            continue
        read_time = time.perf_counter()

        # resize image
        img = cv2.resize(img, target_size)
        resize_time = time.perf_counter()

        # preprocess
        tensor, ratio, padding = yolo.preprocess(img, method="cuda", no_copy=True)
        preprocess_time = time.perf_counter()

        # inference
        outputs = yolo.run(tensor, ratio, padding, preprocessed=True, postprocess=True, no_copy=True)
        inference_time = time.perf_counter()

        # postprocess
        detections = yolo.get_detections(outputs)
        detection_time = time.perf_counter()

        # memory usage in MB
        mem_usage_mb = process.memory_info().rss / (1024 * 1024)

        # log per-frame timing and memory to CSV
        with open(csv_log, "a") as f:
            f.write(f"{frame_id},{args.width},{args.height},"
                    f"{read_time - frame_start:.4f},"
                    f"{resize_time - read_time:.4f},"
                    f"{preprocess_time - resize_time:.4f},"
                    f"{inference_time - preprocess_time:.4f},"
                    f"{detection_time - inference_time:.4f},"
                    f"{detection_time - frame_start:.4f},"
                    f"{mem_usage_mb:.2f}\n")

        # save detections
        frame_detections = []
        if detections is not None:
            for det in detections:
                bbox, confidence, class_id = det
                x1, y1, x2, y2 = bbox
                if confidence >= 0.5:
                    frame_detections.append([int(x1), int(y1), int(x2), int(y2), confidence, int(class_id)])

        if frame_detections:
            output_file_path = os.path.join(output_dir, f"{frame_id:06d}.txt")
            with open(output_file_path, 'w') as f:
                for det in frame_detections:
                    f.write(" ".join(map(str, det)) + "\n")

print(f"Detections saved in {output_dir}/")
print(f"Timing and memory stats saved to {csv_log}")

