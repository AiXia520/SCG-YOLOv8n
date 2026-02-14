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

🚁 Intelligent UAV-Based Agricultural Monitoring and Detection System

With the rapid development of artificial intelligence, computer vision, and unmanned aerial vehicle (UAV) technologies, smart agriculture is undergoing a profound transformation toward precision, automation, and data-driven decision-making. Our ongoing research focuses on the development of an intelligent UAV-based agricultural detection and monitoring system designed to support large-scale crop analysis in complex farmland environments.

The core objective of this project is to leverage drone platforms equipped with high-resolution sensors and deep learning models to enable real-time, accurate, and scalable agricultural perception. By integrating aerial imaging, intelligent recognition algorithms, and edge-cloud collaborative processing, we aim to build a comprehensive UAV-enabled smart agriculture solution capable of supporting crop growth assessment, disease monitoring, yield estimation, and management decision-making.

🌱 Key Research Applications

Our current development focuses on several representative agricultural scenarios:

1. Potato Detection and Counting
We are designing computer vision models capable of accurately detecting potato plants and tuber distributions from UAV aerial images. These models aim to automatically estimate crop density, growth status, and potential yield across large farmland areas, significantly reducing manual labor and improving monitoring efficiency.

2. Potato Leaf Disease and Pest Identification
Early-stage disease and pest detection is critical for reducing crop loss. Using deep learning-based image classification and segmentation techniques, our system identifies common potato leaf diseases and pest damage patterns from UAV imagery, enabling rapid intervention and precision treatment.

3. Tree Crop Detection (Walnut, Camellia Oil Tree, etc.)
In addition to field crops, we are extending our detection framework to orchard and economic forest scenarios, including walnut trees and oil tea (Camellia oleifera) plantations. The system automatically recognizes individual trees, monitors canopy health, estimates plant distribution, and supports digital orchard management.

🧠 System Architecture

The proposed intelligent agricultural detection system consists of four major components:

UAV Data Acquisition Layer
High-resolution RGB and optional multispectral cameras mounted on drones collect real-time aerial imagery across farmland areas under varying environmental conditions.

Edge Intelligence Processing Layer
Lightweight deep learning models deployed on edge devices perform rapid on-site detection, filtering, and preliminary analysis, enabling low-latency responses in the field.

Cloud-Based Model Training and Optimization
Large-scale annotated agricultural datasets are used to continuously train, refine, and evaluate detection models, improving robustness across seasons, crop varieties, and geographic conditions.

Smart Agricultural Visualization Platform
A user-friendly interface displays detection results, crop health maps, disease risk alerts, and statistical reports, supporting farmers and agricultural managers in making informed decisions.

🔬 Research Contributions

This project emphasizes both applied system development and methodological innovation, including:

UAV-based multi-scale crop detection under complex backgrounds

Robust disease recognition under varying illumination and occlusion conditions

Lightweight deep learning models for edge deployment

Data-efficient training strategies for agricultural scenarios with limited labeled data

Integrated perception-to-decision agricultural pipelines

🌍 Practical Impact

By automating crop monitoring tasks traditionally performed through manual inspection, our UAV-based system aims to:

Reduce labor costs and inspection time

Improve early disease detection and prevention

Enable precise yield estimation

Support data-driven agricultural management

Increase overall crop productivity and sustainability

This work is particularly valuable for large-scale farmlands, mountainous agricultural regions, and areas with limited access to traditional monitoring infrastructure.

🚀 Project Status

The intelligent UAV agricultural detection system is currently under active development. Core detection models, data collection workflows, and system integration modules are being continuously improved and tested in real agricultural environments. Future work will expand supported crop types, enhance multi-sensor fusion capabilities, and incorporate real-time decision support functions.

We welcome collaboration from researchers, developers, and agricultural practitioners interested in UAV vision, precision agriculture, and intelligent perception systems.


