"""Export trained model to ONNX with dynamic shapes and run a quick correctness check."""
import argparse
from ultralytics import YOLO
import onnx
import onnxruntime as ort
import numpy as np


def get_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--weights', type=str, required=True)
    ap.add_argument('--onnx', type=str, default='scg_yolov9n.onnx')
    ap.add_argument('--imgsz', type=int, default=640)
    return ap.parse_args()


def main():
    a = get_args()
    model = YOLO(a.weights)
    model.export(format='onnx', dynamic=True, imgsz=a.imgsz, opset=12, simplify=True, file=a.onnx)

    onnx_model = onnx.load(a.onnx)
    onnx.checker.check_model(onnx_model)

    sess = ort.InferenceSession(a.onnx, providers=['CPUExecutionProvider'])
    dummy = np.random.rand(1, 3, a.imgsz, a.imgsz).astype(np.float32)
    outs = sess.run(None, {sess.get_inputs()[0].name: dummy})
    print([o.shape for o in outs])

if __name__ == '__main__':
    main()