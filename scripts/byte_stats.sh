#!/usr/bin/env bash

# -------- CONFIG --------
TRACKER_SCRIPT="please_byte.py"   # tracking script
MEAS_DIR="measurements"                # output folder
NUM_FRAMES_FILE="$MEAS_DIR/num_frames.txt" 
USE_TEGRASTATS=true                   
# ------------------------

mkdir -p "$MEAS_DIR"

# retrieve number of frames
if [ -f "$NUM_FRAMES_FILE" ]; then
    num_frames=$(cat "$NUM_FRAMES_FILE")
else
    # otherwise count number of files in detection folder
    detection_dir="yolox_detections/MOT20-05"
    num_frames=$(ls "$detection_dir" | wc -l)
    echo "$num_frames" > "$NUM_FRAMES_FILE"
fi

echo "[INFO] Measuring tracker performance over $num_frames frames"

# --- FPS Timing ---
echo "[INFO] Measuring FPS..."
track_time=$( /usr/bin/time -f "%e" python3 "$TRACKER_SCRIPT" 2>&1 1>/dev/null )
fps=$(bc -l <<< "$num_frames / $track_time")
echo "$fps" > "$MEAS_DIR/tracker_fps.txt"
echo "[RESULT] FPS: $fps (saved to $MEAS_DIR/tracker_fps.txt)"

# --- Optional: Tegra Power Stats ---
if [ "$USE_TEGRASTATS" = true ]; then
    echo "[INFO] Starting tegrastats for tracker..."
    tegra_log="$MEAS_DIR/tracker_tegrastats.txt"
    rm -f "$tegra_log"
    sudo tegrastats --interval 100 --logfile "$tegra_log" &  # run in background
    TEGRA_PID=$!
    
    python3 "$TRACKER_SCRIPT"
    
    kill $TEGRA_PID
    echo "[RESULT] Tegra power stats saved to $tegra_log"
fi

echo "[DONE] All tracker measurements complete."

