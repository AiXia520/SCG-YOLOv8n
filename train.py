"""Two‑stage training: (1) VisDrone pretrain → (2) Potato fine‑tune
Usage:
# Stage 1 (pretrain)
python train.py --data configs/data_visdrone.yaml --epochs 100 --img 640 --name scg_pre
# Stage 2 (finetune)
python train.py --data configs/data_potato.yaml --epochs 300 --img 640 --name scg_ft --weights runs/detect/scg_pre/weights/best.pt
"""
import argparse
from ultralytics import YOLO




def get_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', type=str, required=True)
    ap.add_argument('--epochs', type=int, default=300)
    ap.add_argument('--img', type=int, default=640)
    ap.add_argument('--batch', type=int, default=32)
    ap.add_argument('--name', type=str, default='scg')
    ap.add_argument('--weights', type=str, default='')
    ap.add_argument('--model', type=str, default='configs/model_scg_yolov9n.yaml')
    return ap.parse_args()




def main():
    args = get_args()
    model = YOLO(args.model if not args.weights else args.weights)
    model.train(
    data=args.data,
    epochs=args.epochs,
    imgsz=args.img,
    batch=args.batch,
    name=args.name,
    device=0,
    optimizer='SGD',
    lr0=0.01,
    momentum=0.937,
    weight_decay=5e-4,
    close_mosaic=10,
    cos_lr=False,
    )


if __name__ == '__main__':
    main()