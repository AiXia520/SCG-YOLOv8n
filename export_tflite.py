import argparse, os, shutil, subprocess

ap = argparse.ArgumentParser()
ap.add_argument('--onnx', type=str, default='scg_yolov9n.onnx')
ap.add_argument('--outdir', type=str, default='tflite_build')
ap.add_argument('--float16', action='store_true')
args = ap.parse_args()

os.makedirs(args.outdir, exist_ok=True)
# 1) ONNX→TF SavedModel
subprocess.check_call([
    'onnx2tf', '-i', args.onnx, '-o', args.outdir, '--output_signaturedefs',
])
# 2) TF SavedModel→TFLite
import tensorflow as tf
converter = tf.lite.TFLiteConverter.from_saved_model(args.outdir)
if args.float16:
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_types = [tf.float16]

tflite_model = converter.convert()
with open(os.path.join(args.outdir, 'scg_yolov9n_fp16.tflite' if args.float16 else 'scg_yolov9n_fp32.tflite'), 'wb') as f:
    f.write(tflite_model)
print('TFLite written to', args.outdir)