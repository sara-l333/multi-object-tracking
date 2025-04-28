#!/usr/bin/env bash

# -------- CONFIG --------
SEQUENCE_DIR="../../MOT20/train/MOT20-05"  # original sequence directory (TODO change per video)
SEQUENCE_NAME=$(basename "$SEQUENCE_DIR")
RESIZE_SCRIPT="resize_images.py"
WIDTH=1654
HEIGHT=1080
RESIZED_DIR="resized_${WIDTH}x${HEIGHT}/${SEQUENCE_NAME}"
DETECTOR_SCRIPT="please_yolox.py"
MEAS_DIR="measurements"
USE_PROFILER=false
USE_TEGRASTATS=true
# ------------------------

mkdir -p "$MEAS_DIR"
mkdir -p "$RESIZED_DIR"

# resize images ahead of time
echo "[INFO] Resizing images to $WIDTH x $HEIGHT..."
python3 "$RESIZE_SCRIPT" \
  --src "$SEQUENCE_DIR/img1" \
  --dst "$RESIZED_DIR" \
  --width "$WIDTH" \
  --height "$HEIGHT"

# get number of frames from seqinfo.ini (included in MOT20 video directory)
seqinfo_path="$SEQUENCE_DIR/seqinfo.ini"
num_frames=$(sed '5q;d' "$seqinfo_path")
num_frames=${num_frames#*=}
echo "$num_frames" > "$MEAS_DIR/num_frames.txt"

echo "[INFO] Measuring detector performance over $num_frames frames"

# --- FPS Timing ---
echo "[INFO] Measuring FPS..."
detect_time=$( /usr/bin/time -f "%e" python3 "$DETECTOR_SCRIPT" \
  --input_dir "$RESIZED_DIR" 2>&1 1>/dev/null )
fps=$(bc -l <<< "$num_frames / $detect_time")
echo "$fps" > "$MEAS_DIR/detector_fps.txt"
echo "[RESULT] FPS: $fps (saved to $MEAS_DIR/detector_fps.txt)"

# --- Tegra Power Stats ---
if [ "$USE_TEGRASTATS" = true ]; then
    echo "[INFO] Starting sudo tegrastats for detector..."
    tegra_log="$MEAS_DIR/detector_tegrastats.txt"
    rm -f "$tegra_log"
    
    # run tegrastats with sudo, logging output to a file
    sudo tegrastats --interval 100 --logfile "$tegra_log" &
    TEGRA_PID=$!

    python3 "$DETECTOR_SCRIPT" --input_dir "$RESIZED_DIR" --width "$WIDTH" --height "$HEIGHT"

    # stop tegrastats logging once the detector is done
    kill $TEGRA_PID
    echo "[RESULT] Tegra power stats saved to $tegra_log"
fi

echo "[DONE] All measurements complete."

