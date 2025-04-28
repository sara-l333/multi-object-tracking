import os
import numpy as np
from boxmot.trackers.bytetrack.bytetrack import ByteTrack

# tracking params
tracker = ByteTrack(track_thresh=0.5, match_thresh=0.8, track_buffer=30, frame_rate=30)

# path to detection results folder
detections_folder = "yolox_detections/MOT20-05"

# TODO: replace with actual image dimensions
img_height = 1080
img_width = 1654

# create dummy image array (black image)
dummy_image = np.zeros((img_height, img_width, 3), dtype=np.uint8)

# list of detection files sorted by frame number
detection_files = sorted(os.listdir(detections_folder))

for fname in detection_files:
    frame_id = int(os.path.splitext(fname)[0])
    fpath = os.path.join(detections_folder, fname)
    detections = []
    with open(fpath, "r") as f:
        for line in f:
            values = list(map(float, line.strip().split()))
            if len(values) == 6:  # YOLOX output: x1, y1, x2, y2, score, class_id
                detections.append(values)
            else:
                print(f"Warning: Skipping malformed line in {fname}: {line.strip()}")
    if detections:
        det = np.array(detections)  # convert to numpy array
    else:
        print("Warning: No detections found, using empty array.")
        det = np.empty((0, 6))  # empty array with shape (0, 6)
    
    # ensure det is a numpy array with the correct shape (n, 6)
    if det.ndim != 2 or det.shape[1] != 6:
        print(f"Error: Detection array has an invalid shape: {det.shape}")
        continue  # skip this frame
    
    # ensure that det is passed as a numpy array, even if it's empty
    if isinstance(det, list):
        det = np.array(det)
    
    # call ByteTrack update
    # use the dummy image or replace it with your actual frame image if available
    online_targets = tracker.update(det, dummy_image)
    
    # print frame number, ID for detected object, bounding box dimensions
    # [x1, y1, width, height], and confidence score that object is identified
    # correctly
    output_file = "tracking_results.txt"
    with open(output_file, "a") as f:
        for t in online_targets:
            tid = int(t[4])
            tlwh = [t[0], t[1], t[2] - t[0], t[3] - t[1]]  # convert xyxy to tlwh
            score = float(t[5])
            # Write the results for the frame to the file
            f.write(f"Frame {frame_id}: ID {tid} at {tlwh} with score {score:.2f}\n")

print(f"Results written to {output_file}")

