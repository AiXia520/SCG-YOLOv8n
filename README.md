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

In the next step, we will base our research on drones to explore the application of intelligent drones in the agricultural field, including the detection and counting of potatoes, the identification of diseases and pests on potato leaves, and the detection of related crops such as walnuts and oil tea. We will further develop an intelligent agricultural detection system based on drones, which is currently under development.

**Latest News**
**Task 1: Potato Quality Classification and Defect Detection**

As the next stage of our UAV-enabled intelligent agriculture research, we are developing an automated potato quality inspection and grading system based on deep learning and computer vision techniques. This task focuses on fine-grained classification and detection of potato conditions to support post-harvest sorting and quality control.

The system is designed to recognize and categorize potatoes into five key classes: damaged potatoes, unqualified potatoes, fungal-infected potatoes, healthy potatoes, and sprouted potatoes. By training convolutional neural networks and object detection models on diverse potato image datasets collected under real agricultural and industrial conditions, the system aims to achieve high accuracy and robustness across variations in lighting, size, texture, and surface contamination.

Through real-time visual analysis, the proposed approach enables rapid identification of defective or diseased potatoes, helping reduce waste, improve food safety, and increase the efficiency of grading and sorting processes. This intelligent classification module will be integrated into our broader smart agriculture pipeline, supporting both UAV-based field monitoring and ground-level post-harvest inspection scenarios.

<img width="1842" height="1093" alt="image" src="https://github.com/user-attachments/assets/f85571f5-dce7-4bc2-a549-874931aea1ac" />
<img width="1523" height="1002" alt="image" src="https://github.com/user-attachments/assets/bf15741e-b599-4732-a09e-4fca52dfa7d0" />

**Task 2: Intelligent UAV-Based Agricultural Monitoring and Detection System**

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




