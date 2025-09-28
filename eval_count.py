"""Count potatoes per image from detections and compute RMSE / MAPE / R^2.
Assumes a single class named 'potato'.


Usage:
python eval_count.py --weights runs/detect/scg_ft/weights/best.pt --images data/potato/images/test
"""
import argparse
import glob
import os
from ultralytics import YOLO
import cv2


from scg_yolov9n.counting import rmse, mape, r2


def get_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--weights', type=str, required=True)
    ap.add_argument('--images', type=str, required=True)
    ap.add_argument('--conf', type=float, default=0.25)
    return ap.parse_args()


def main():
    args = get_args()
    model = YOLO(args.weights)
    image_paths = sorted(glob.glob(os.path.join(args.images, '*.*')))
    y_true, y_pred = [], []


# Expect a parallel txt with ground truth counts, e.g., IMG_0001.jpg → IMG_0001.txt containing a single integer
for p in image_paths:
    res = model.predict(p, conf=args.conf, verbose=False)[0]
    pred_count = (res.boxes.cls == 0).sum().item() if res.boxes is not None else 0
    y_pred.append(pred_count)
    gt_txt = os.path.splitext(p)[0] + '.txt'
    if os.path.exists(gt_txt):
        with open(gt_txt, 'r') as f:
            y_true.append(int(f.read().strip()))
    else:
        # if no GT, skip metric aggregation for this image
        y_pred.pop()

    if y_true:
        print(f"RMSE: {rmse(y_true, y_pred):.3f}")
        print(f"MAPE: {mape(y_true, y_pred):.2f}%")
        print(f"R^2 : {r2(y_true, y_pred):.3f}")
    else:
        print("No ground‑truth count files found. Only produced predictions.")


if __name__ == '__main__':
    main()

