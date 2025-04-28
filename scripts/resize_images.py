import os
import cv2
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--src', type=str, required=True, help='Path to source image folder')
parser.add_argument('--dst', type=str, required=True, help='Path to destination resized image folder')
parser.add_argument('--width', type=int, required=True, help='Target width')
parser.add_argument('--height', type=int, required=True, help='Target height')
args = parser.parse_args()

target_size = (args.width, args.height)
os.makedirs(args.dst, exist_ok=True)

for img_name in sorted(os.listdir(args.src)):
    if img_name.endswith(".jpg"):
        src_path = os.path.join(args.src, img_name)
        dst_path = os.path.join(args.dst, img_name)

        img = cv2.imread(src_path)
        if img is None:
            print(f"Warning: Failed to read {img_name}")
            continue

        resized = cv2.resize(img, target_size)
        cv2.imwrite(dst_path, resized)

print(f"[DONE] Resized images saved to: {args.dst}")

