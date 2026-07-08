# Garbage Vulnerability Point (GVP) Monitoring System

## Overview
The **Garbage Vulnerability Point (GVP) Monitoring System** is a comprehensive solution designed to monitor, segment, and analyze waste and garbage vulnerability points using computer vision and edge computing. The system utilizes deep learning segmentation models (such as YOLOv11, DDRNet23, FCN-ResNet50, and UNet) to identify and classify objects like **Waste**, **Animals**, and **Persons** in real-time or from captured images.

The project is divided into several modules, targeting different platforms including Android (for mobile on-device inference), Raspberry Pi (for edge-based web monitoring), and desktop (for model evaluation and sensor data visualization).

## Project Structure

- **`Code/`**: Contains all the source code for the project, divided into several sub-components:
  - **`QIDK_app_code/`**: An Android application built with Kotlin/Java that performs on-device waste segmentation using TensorFlow Lite (`yolo11_seg.tflite`). It captures images, processes them, calculates per-class pixel coverage, and can upload the results and statistics to Firebase. It also contains `GUI.py`, a desktop serial interface for reading and plotting sensor data (e.g., temperature and humidity) via a COM port.
  - **`raspi.py`**: A Flask-based web application intended for deployment on a Raspberry Pi. It provides a web interface to upload images and uses Ultralytics YOLO to perform segmentation, returning the segmented image along with coverage statistics and inference time.
  - **`gui/run_all_models_gui.py`**: A unified Tkinter-based desktop application that allows developers to load multiple segmentation models simultaneously (such as DDRNet23, FCN-ResNet50, YOLOv8/v11, UNet) and run inference on a single image to compare their performance and outputs.
- **`Demos/`**: Contains demonstration videos and images showcasing the system in action.
- **`Presentation/`**: Contains the PDF presentations used for project evaluation and overview.
- **`Report/`**: Contains detailed documentation and the final project report (`GVP_report.pdf`), as well as specific setup instructions for the Android application.

## Features

1. **On-Device Mobile Segmentation**: The Android app performs fast, on-device segmentation using a quantized TFLite model. It supports auto-capture intervals and cloud synchronization with Firebase Realtime Database and Storage.
2. **Edge Computing via Raspberry Pi**: The Flask app (`raspi.py`) enables headless or lightweight edge devices to serve a web interface for remote image uploading and segmentation using Python-based YOLO models.
3. **Multi-Model Evaluation GUI**: The Python desktop GUI allows researchers to compare the efficacy of different segmentation architectures side-by-side, analyzing coverage percentages and output masks.
4. **Sensor Data Integration**: The `GUI.py` script facilitates reading serial data (like temperature and humidity) from connected microcontrollers to supplement visual data with environmental metrics.

## Getting Started

### Android App
1. Open the `Code/QIDK_app_code` folder in Android Studio.
2. Ensure you have the `yolo11_seg.tflite` model in the `app/src/main/assets/` directory.
3. (Optional) Configure Firebase by adding your `google-services.json`.
4. Build and run on an Android device (ARM64 recommended).

### Raspberry Pi Web App
1. Navigate to the `Code` directory.
2. Place your `yolov11.pt` model file in the directory.
3. Run `python3 raspi.py` to start the Flask server.
4. Access the web interface via the device's IP address.

### Multi-Model Evaluation GUI
1. Navigate to `Code/gui`.
2. Install requirements using `pip install -r requirements.txt`.
3. Ensure model files (`.pt` or `.pth`) are placed in the `./models/` directory.
4. Run `python3 run_all_models_gui.py`.

*Note: Large model files (`.pt` and `.tflite` model weights) are excluded from this repository due to size constraints.*

## Contributors

- [Phanindra Gollapalli](https://github.com/phanindragollapalli)
- [Krithik Kambhampati](https://github.com/krithikkambhampati)
- [Krishna Kaundinya Peddibhotla](https://github.com/Kaundinya-P)
- [Nikhith Reddy Kummathi](https://github.com/Nikhith-2025)
