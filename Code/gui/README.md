Multi-Model Segmentation GUI

Overview
- This GUI (run_all_models_gui.py) attempts to load several segmentation/detection models and lets you run the same image through all loaded models.
- Place model files inside the `models/` folder (created next to the repository root). The GUI will try to auto-load files it finds.

Expected model filenames (put these in `./models/`):
- ddrnet23_slim_best.pt    -> TorchScript for DDRNet23
- fcn_resnet50_best.pt     -> TorchScript for FCN-ResNet50
- yolov8_best.pt           -> YOLOv8 model (Ultralytics .pt)
- yolov11_gvp_new.pt       -> YOLOv11 model (Ultralytics-compatible)
- unet_garbage_multiclass.pt -> UNet state_dict (.pt/.pth)

Notes and recommendations
- TorchScript (.pt created by torch.jit.save) is the most robust for loading without class definitions.
If a model is saved as a state_dict (common .pth/.pt), the script includes a compact UNet definition and will attempt to load the state_dict into that class. If your model uses a very different architecture, loading may fail.
- YOLO models should be Ultralytics-compatible and can be loaded using `from ultralytics import YOLO`.

How to run
1. Create a python environment (recommended) and install the requirements listed in `requirements.txt`.

   pip install -r gui/requirements.txt

2. Put your model files into `./models/` (create the folder if it does not exist). Use the filenames suggested above or update the script to point to your filenames.
3. Run the GUI:

   python gui/run_all_models_gui.py

4. Use "Select Image" to pick an image. Click "Run All Models" to run the same image across all models that loaded successfully.

What the GUI shows
- A tab per expected model. Tabs for models that failed to load will still exist but will show an error message in the right panel.
- For each model you will see the overlayed result and a small "masks only" thumbnail plus a coverage percentage (segmented pixels / total pixels).

If something fails
- Look at the terminal output for tracebacks.
- Common problems:
  - Missing dependencies (install packages in `requirements.txt`).
  - Model incompatible (not TorchScript nor compatible state dict). Consider exporting to TorchScript or using the exact architecture definition used in training.

If you want, I can:
- Wire each model to the exact training code's preprocessing/postprocessing (recommended for accuracy).
- Add a settings panel to change image preprocess sizes and thresholds.
- Save per-model outputs automatically to a chosen folder.
