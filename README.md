ultralytics>=8.3.0
opencv-python
numpy
pandas
matplotlib
pyyaml
scipy
onnx
onnxruntime
onnxsim
tensorflow-cpu==2.15.* # for tflite conversion; use -gpu if needed
onnx2tf>=1.21.0 # optional: alternative ONNX→TFLite path


Android quick start (CameraX + TFLite)
Create a new Android Studio project (Empty Activity). Min SDK 24+, Kotlin.

Add dependencies in app/build.gradle:

implementation 'org.tensorflow:tensorflow-lite:2.15.0'
implementation 'org.tensorflow:tensorflow-lite-gpu:2.15.0'  // optional
implementation 'androidx.camera:camera-core:1.3.4'
implementation 'androidx.camera:camera-camera2:1.3.4'
implementation 'androidx.camera:camera-lifecycle:1.3.4'
implementation 'androidx.camera:camera-view:1.3.4'
Put scg_yolov9n_fp16.tflite into app/src/main/ml/ and a labels.txt with one line: potato.

Use a standard TFLite detector wrapper to feed YUV_420_888 frames (via CameraX) → resize to 640×640 → run → decode boxes (NMS) → draw overlays.

On mid‑range devices you should see real‑time FPS; prefer FP16. Enable NNAPI if available.
