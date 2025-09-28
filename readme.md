# SCG-YOLOv9n PotatoDetector

## A compact, mobile‑deployable potato detection & counting framework based on YOLOv9n with three lightweight modules:

## C‑SPD (Contextual Space‑to‑Depth) for spatial‑detail retention

## S‑CARAFE for semantic‑aware upsampling

## GhostShuffleConv for efficient expressiveness in the head

This repo includes training (VisDrone pretrain → potato fine‑tune), counting metrics (RMSE / MAPE), and export to ONNX → TFLite (float16). An Android skeleton and guidance are provided for on‑device inference.

scg-yolov9n/
├─ README.md
├─ requirements.txt
├─ configs/
│ ├─ data_potato.yaml
│ ├─ data_visdrone.yaml
│ └─ model_scg_yolov9n.yaml
├─ scg_yolov9n/
│ ├─ __init__.py
│ ├─ modules.py # C-SPD, S-CARAFE, GhostShuffleConv
│ ├─ model.py # Model builder: inject modules into YOLOv9n graph
│ ├─ augment.py # (optional) extra augmentations inc. synthetic occlusion
│ └─ counting.py # RMSE, MAPE, R^2 utilities
├─ train.py # stage-1 pretrain (VisDrone), stage-2 finetune (potato)
├─ eval_count.py # evaluate counting metrics from detections
├─ export_onnx.py # export to ONNX (+ dynamic shapes) and test
├─ export_tflite.py # convert ONNX → TFLite (float16) locally
└─ android/
├─ README.md # Android setup notes (CameraX + TFLite)
└─ app_skeleton.zip # minimal Android Studio project (placeholder)


# SCG‑YOLOv9n PotatoDetector

A lightweight, field‑tested pipeline for potato detection & counting with mobile deployment.

## Highlights
- **C‑SPD** retains fine‑grained spatial detail for small/occluded tubers.
- **S‑CARAFE** improves boundary recovery during upsampling.
- **GhostShuffleConv** offers efficient, expressive heads.
- **Two‑stage training**: VisDrone pretrain → potato fine‑tune.
- **Mobile**: ONNX→TFLite FP16, Android (CameraX) demo.

## Install
```bash
conda create -n scg python=3.10 -y
conda activate scg
pip install -r requirements.txt
```


## Data

Prepare VisDrone2019‑DET and your potato dataset in YOLO format.

Edit paths in configs/data_*.yaml.


## Train
# Stage 1: pretrain on VisDrone (optional but recommended)
python train.py --data configs/data_visdrone.yaml --epochs 100 --img 640 --name scg_pre

# Stage 2: fine‑tune on potato
python train.py --data configs/data_potato.yaml --epochs 300 --img 640 --name scg_ft \
  --weights runs/detect/scg_pre/weights/best.pt


## Evaluate counting
```bash
python eval_count.py --weights runs/detect/scg_ft/weights/best.pt --images /data/potato/images/test
```


## Export
```bash
python export_onnx.py --weights runs/detect/scg_ft/weights/best.pt --onnx scg_yolov9n.onnx
python export_tflite.py --onnx scg_yolov9n.onnx --float16
```


## Android

See android/README.md. Place the generated *.tflite and labels.txt into the project and run on device.



