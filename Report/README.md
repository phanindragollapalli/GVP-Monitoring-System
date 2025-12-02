# ESW Waste Segmentation

## Description

This Android application performs on-device waste segmentation using a TensorFlow Lite model trained to detect classes such as Animal, Waste, and Person. Images are captured from the camera or selected from the gallery, segmented on the device, and the results (masks and statistics) are saved locally and optionally uploaded to Firebase for further analysis.

## Getting Started

To use this repository for segmentation on your own device or model, follow these steps:

1. Clone the repository to your local machine.
2. Open the ESW folder as a project in Android Studio.
3. Ensure you have Android SDK 34 installed and a device with Android 9 (API 28)+ and ARM64 (arm64-v8a) CPU.
4. Verify the segmentation model:
   - Default model file: app/src/main/assets/yolo11_seg.tflite.
   - If you use another model, replace this file and keep the same name, or update the path in SegmentationHelper.kt.
5. Configure Firebase (optional but recommended):
   - Create a Firebase project and an Android app with package name com.example.esw.
   - Download google-services.json and place it in the ESW/app/ directory.
   - Make sure the Realtime Database URL in MainActivity.kt matches your Firebase project.
6. Let Gradle sync and resolve all dependencies.
7. Connect an Android device (USB debugging enabled) or start an ARM64 emulator.
8. Select the app configuration in Android Studio and click Run.

## How the Model Is Integrated

The TensorFlow Lite model file yolo11_seg.tflite is loaded from the assets folder using a memory-mapped ByteBuffer.

Before inference, each image is converted to ARGB_8888, resized to 640×640 with letterboxing, and normalized to [0, 1].

The model outputs bounding boxes, class scores, and segmentation masks.

Post-processing builds per-class masks, overlays them on the original image, and computes:
- Per-class pixel counts and coverage percentage.
- Overall image coverage and timing statistics (preprocess, inference, postprocess).

All of this logic is implemented in SegmentationHelper.kt and used by MainActivity.kt when you capture or select an image.

## Where Is Collected Data Stored?

### On the device (local files)

**Folder:** Android/data/com.example.esw/files/segmentation_captures/.

**Files:**
- captures.csv – one line per capture with timestamps, coverage, delegate used, and local image path.
- seg_<timestamp>.jpg – segmented output images.

### In Firebase (if configured)

**Realtime Database node:** segmentation_results.
- Each entry contains timestamp, coverage, processing times, class breakdown, and (if available) the image URL.

**Firebase Storage folder:** segmented/seg_<timestamp>.jpg for uploaded images.

## How to Download This Collected Data

### Local files from device

In Android Studio, open Device File Explorer and navigate to Android/data/com.example.esw/files/segmentation_captures/.

Right-click the folder or files and choose Save As to download them.

Alternatively, use adb pull on that folder.

### From Firebase

Use the Firebase Console to export data from the segmentation_results node as JSON.

Download images directly from the Firebase Storage browser or via the Storage SDK / CLI.

## How to Use Debug Logging

Open Logcat in Android Studio while the app is running.

Filter by tags:
- MainActivity – camera, capture flow, Firebase logging.
- SegmentationHelper – model loading, delegate selection, timing, and post-processing.

You can add your own logs using Log.d, Log.i, or Log.e in the Kotlin files.

## How to Change Auto-Capture Interval

In the app, tap Start Auto Capture and enter the desired interval in seconds.

To change the default value in code, adjust the logic in MainActivity.kt inside showAutoCaptureDialog() and the autoCaptureIntervalMs variable.

## Minimal Requirements

- Android Studio with Gradle plugin 8.x.
- Android device with Android 9+ and ARM64 (arm64-v8a).
- Internet access if you want to use Firebase upload.
- For the optional desktop serial GUI (ESW/GUI.py): Python 3.8+, pyserial, and matplotlib.
