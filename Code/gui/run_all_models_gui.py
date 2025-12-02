"""
Unified Multi-Model Segmentation GUI
- Places to put model files: ./models/
- Names expected (see README.md for details)

This GUI will try to load the common models used across the repo:
 - DDRNet23 (TorchScript .pt)
This GUI will try to load the common models used across the repo:
 - DDRNet23 (TorchScript .pt)
 - FCN-ResNet50 (TorchScript .pt)
 - YOLOv8 / YOLOv11 (Ultralytics .pt)
 - UNet (PyTorch .pt/pth state_dict)

Behavior:
 - Create a tab per model (if model file exists)
 - Select an image and "Run All" to run the same image through all loaded models
 - Shows overlay result per-model and a "masks only" thumbnail plus coverage percent

Notes / assumptions:
 - TorchScript .pt files should be saved with torch.jit.save or be compatible with torch.jit.load
 - If a model is a state_dict, the script will attempt to build a compatible architecture (UNet/SAM/Custom) and load_state_dict
 - YOLO models are loaded via ultralytics.YOLO and must be compatible with that package

This is a convenience front-end. If any model requires special preprocessing / postprocessing not captured here, we fall back to best-effort inference.
"""

import os
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk
import numpy as np
import cv2

try:
    import torch
    import torch.nn as nn
    from torchvision import transforms
except Exception:
    torch = None

try:
    from ultralytics import YOLO
except Exception:
    YOLO = None

# ----------------- Minimal helper model classes ----------------- #
# Re-declare compact versions of the known model classes used in the repo
class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=4):
        super(UNet, self).__init__()
        def conv_block(in_c, out_c):
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, 3, padding=1),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_c, out_c, 3, padding=1),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True)
            )
        self.enc1 = conv_block(in_channels, 64)
        self.enc2 = conv_block(64, 128)
        self.enc3 = conv_block(128, 256)
        self.enc4 = conv_block(256, 512)
        self.pool = nn.MaxPool2d(2)
        self.center = conv_block(512, 1024)
        self.up4 = nn.ConvTranspose2d(1024, 512, 2, 2)
        self.dec4 = conv_block(1024, 512)
        self.up3 = nn.ConvTranspose2d(512, 256, 2, 2)
        self.dec3 = conv_block(512, 256)
        self.up2 = nn.ConvTranspose2d(256, 128, 2, 2)
        self.dec2 = conv_block(256, 128)
        self.up1 = nn.ConvTranspose2d(128, 64, 2, 2)
        self.dec1 = conv_block(128, 64)
        self.out = nn.Conv2d(64, out_channels, 1)
    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        c = self.center(self.pool(e4))
        d4 = self.dec4(torch.cat([self.up4(c), e4], 1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], 1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], 1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], 1))
        return self.out(d1)

class CustomSegModel(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 64, 7, 2, 3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.seg_head = nn.Sequential(
            nn.Conv2d(64, 32, 3, 1, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, num_classes, 1),
        )
    def forward(self, x):
        features = self.backbone(x)
        decoded = self.decoder(features)
        decoded = torch.nn.functional.interpolate(decoded, size=x.shape[-2:], mode='bilinear', align_corners=False)
        out = self.seg_head(decoded)
        return out

# ----------------- GUI ----------------- #
MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
MODEL_DIR = os.path.abspath(MODEL_DIR)
os.makedirs(MODEL_DIR, exist_ok=True)

DEFAULT_MODEL_FILES = {
    'DDRNet23': os.path.join(MODEL_DIR, 'ddrnet23_slim_best.pt'),
    'FCN-ResNet50': os.path.join(MODEL_DIR, 'fcn_resnet50_best.pt'),
    'YOLOv8': os.path.join(MODEL_DIR, 'yolov8_best.pt'),
    'YOLOv11': os.path.join(MODEL_DIR, 'yolov11_gvp_new.pt'),
    'UNet': os.path.join(MODEL_DIR, 'unet_garbage_multiclass.pt')
}

# Consistent class colors (RGB) and ordering
CLASS_COLORS = {
    'animal': (34, 139, 34),    # forest green
    'waste':  (255, 165, 0),    # orange
    'person': (30, 144, 255),   # dodger blue
}

# Order and display names for the main classes we care about
# show Waste first in the table, then Animal, then Person
CLASS_ORDER = ['waste', 'animal', 'person']
CLASS_DISPLAY = {'animal': 'Animal', 'waste': 'Waste', 'person': 'Person'}


def make_mask_preview(mask_rgb, out_size=(280, 180), bg=(0, 0, 0)):
    """Return a PIL.Image showing masks-only view with black background.
    Scales the full mask to fit the preview area while preserving aspect ratio.
    """
    from PIL import Image
    if mask_rgb is None:
        return Image.new('RGB', out_size, bg)

    try:
        # Ensure mask is uint8 and in correct format
        if not isinstance(mask_rgb, np.ndarray):
            return Image.new('RGB', out_size, bg)
        
        # Ensure uint8 type
        if mask_rgb.dtype != np.uint8:
            mask_rgb = mask_rgb.astype(np.uint8)
        
        # Get mask dimensions (height, width, channels)
        h, w = mask_rgb.shape[:2]
        if w == 0 or h == 0:
            return Image.new('RGB', out_size, bg)
        
        # Convert numpy array to PIL Image
        pil_mask = Image.fromarray(mask_rgb, mode='RGB')
        
        # Calculate scale to fit within preview while preserving aspect ratio
        scale = min(out_size[0] / w, out_size[1] / h)
        new_w = max(1, int(w * scale))
        new_h = max(1, int(h * scale))
        
        # Resize using NEAREST to preserve mask colors/edges
        pil_mask = pil_mask.resize((new_w, new_h), Image.Resampling.NEAREST)
        
        # Create black canvas and center the mask
        canvas = Image.new('RGB', out_size, bg)
        x = (out_size[0] - new_w) // 2
        y = (out_size[1] - new_h) // 2
        canvas.paste(pil_mask, (x, y))
        
        return canvas
    except Exception as e:
        print(f"Error creating mask preview: {e}")
        import traceback
        traceback.print_exc()
        return Image.new('RGB', out_size, bg)
class MultiModelGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Multi-Model Segmentation Runner")
        self.root.geometry("1400x900")
        self.models = {}
        self.model_widgets = {}
        self.image_path = None
        self.total_pixels = 0
        self.device = torch.device('cuda' if torch and torch.cuda.is_available() else 'cpu') if torch else None

        self.build_ui()
        threading.Thread(target=self.auto_load_models, daemon=True).start()

    def build_ui(self):
        # Top header with cleaner styling
        style = ttk.Style()
        try:
            style.theme_use('clam')
        except Exception:
            pass

        header = tk.Frame(self.root, bg='#22313F')
        header.pack(fill=tk.X)
        title = tk.Label(header, text='🎯 Multi-Model Segmentation Runner', bg='#22313F', fg='white',
                         font=('Segoe UI', 16, 'bold'))
        title.pack(side=tk.LEFT, padx=12, pady=10)

        btn_frame = tk.Frame(header, bg='#22313F')
        btn_frame.pack(side=tk.RIGHT, padx=12)

        select_btn = ttk.Button(btn_frame, text='📁 Select Image', command=self.select_image, width=16)
        select_btn.pack(side=tk.LEFT, padx=6)
        self.run_btn = ttk.Button(btn_frame, text='▶ Run All Models', command=self.run_all, state=tk.DISABLED, width=16)
        self.run_btn.pack(side=tk.LEFT, padx=6)
        open_models_btn = ttk.Button(btn_frame, text='📂 Open Models Folder', command=lambda: os.startfile(MODEL_DIR) if os.name=='nt' else os.system(f'xdg-open "{MODEL_DIR}"'), width=18)
        open_models_btn.pack(side=tk.LEFT, padx=6)

        self.status = tk.Label(self.root, text='Loading models...', anchor='w', bg='#f4f6f7')
        self.status.pack(fill=tk.X)

        self.notebook = ttk.Notebook(self.root, padding=6)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=12, pady=12)

        # Create tabs placeholder for expected models
        for name in DEFAULT_MODEL_FILES.keys():
            frame = tk.Frame(self.notebook, bg='white')
            self.notebook.add(frame, text=name)
            left = tk.Frame(frame, bg='white')
            left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
            label = tk.Label(left, text=f"{name} output will appear here", bg='white', font=('Segoe UI', 11), anchor='center', justify='center')
            label.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
            right = tk.Frame(frame, bg='white', width=300)
            right.pack(side=tk.RIGHT, fill=tk.Y)
            right.pack_propagate(False)
            card = tk.Frame(right, bg='#f7f9fa', relief=tk.RIDGE, bd=1)
            card.pack(padx=12, pady=12, fill=tk.BOTH, expand=False)
            tk.Label(card, text='Mask preview', bg='#f7f9fa', font=('Segoe UI', 10, 'bold')).pack(pady=(8,4))
            mask_thumb = tk.Label(card, text='', bg='#000000', relief=tk.SUNKEN)
            mask_thumb.pack(pady=4, padx=8)
            coverage = tk.Label(card, text='Coverage: N/A', bg='#f7f9fa', font=('Segoe UI', 10))
            coverage.pack(pady=(6, 6))

            # Stats / legend area
            stats_frame = tk.Frame(card, bg='#f7f9fa')
            stats_frame.pack(fill=tk.X, padx=8, pady=(4,8))
            tk.Label(stats_frame, text='Legend / Stats', bg='#f7f9fa', font=('Segoe UI', 9, 'bold')).pack(anchor='w')

            self.model_widgets[name] = {
                'frame': frame,
                'label': label,
                'mask': mask_thumb,
                'coverage': coverage,
                'stats_frame': stats_frame
            }

    def auto_load_models(self):
        found = False
        for name, path in DEFAULT_MODEL_FILES.items():
            if os.path.exists(path):
                ok = self.load_model(name, path)
                if ok:
                    found = True
        self.status.config(text='Models loaded' if found else 'No models found in ./models/ - see README in gui/')

    def load_model(self, model_name, path):
        try:
            self.status.config(text=f'Loading {model_name}...')
            if model_name in ['YOLOv8', 'YOLOv11'] and YOLO is not None:
                model = YOLO(path)
            elif model_name == 'UNet' and torch is not None:
                model = UNet()
                model.load_state_dict(torch.load(path, map_location=self.device))
                model.to(self.device).eval()
            # (SAM2 removed) fall through to generic TorchScript loader below if applicable
            elif torch is not None:
                # try torch.jit.load for DDRNet / FCN
                model = torch.jit.load(path, map_location=self.device)
                model.eval()
            else:
                raise RuntimeError('Torch is not available in environment')

            self.models[model_name] = model
            self.model_widgets[model_name]['coverage'].config(text='Coverage: ready')
            return True
        except Exception as e:
            print(f'Failed to load {model_name} from {path}: {e}')
            self.model_widgets[model_name]['coverage'].config(text='Coverage: model load failed')
            return False

    def select_image(self):
        path = filedialog.askopenfilename(filetypes=[('Images','*.jpg *.jpeg *.png *.bmp *.tiff')])
        if not path:
            return
        self.image_path = path
        img = cv2.imread(path)
        self.total_pixels = img.shape[0] * img.shape[1]
        self.status.config(text=f'Selected {os.path.basename(path)} | {self.total_pixels:,} px')
        if self.models:
            self.run_btn.config(state=tk.NORMAL)

        # show original in each tab quickly
        for name, w in self.model_widgets.items():
            lbl = w['label']
            pil = Image.open(path).convert('RGB')
            pil.thumbnail((800,600), Image.Resampling.LANCZOS)
            imgtk = ImageTk.PhotoImage(pil)
            lbl.config(image=imgtk, text='')
            lbl.image = imgtk

    def run_all(self):
        if not self.image_path:
            messagebox.showwarning('No image', 'Please select an image first')
            return
        threading.Thread(target=self._run_all_thread, daemon=True).start()

    def _run_all_thread(self):
        for name, model in list(self.models.items()):
            try:
                self.status.config(text=f'Running {name}...')
                if name in ['YOLOv8', 'YOLOv11'] and YOLO is not None:
                    res = model(self.image_path)
                    # produce overlay and mask
                    img = cv2.cvtColor(cv2.imread(self.image_path), cv2.COLOR_BGR2RGB)
                    masks = np.zeros_like(img)
                    classes = {}
                    det_counts = {}
                    for r in res:
                        if getattr(r, 'masks', None) is None:
                            continue
                        for i in range(len(r.boxes)):
                            mask = r.masks.data[i].cpu().numpy()
                            mask = cv2.resize(mask, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
                            mask_bin = (mask > 0.5).astype(np.uint8)
                            cls = model.names[int(r.boxes.cls[i])]
                            # consistent color choice based on name
                            cname = cls.lower()
                            color = CLASS_COLORS.get(cname, (int(np.random.randint(60,240)), int(np.random.randint(60,240)), int(np.random.randint(60,240))))
                            masks[mask_bin==1] = color
                            classes[cls] = classes.get(cls, 0) + int(mask_bin.sum())
                            det_counts[cls] = det_counts.get(cls, 0) + 1
                    # Normalize class keys to our main classes (lowercase)
                    pixels_for_class = {k: 0 for k in CLASS_ORDER}
                    counts_for_class = {k: 0 for k in CLASS_ORDER}
                    for k, px in classes.items():
                        lk = k.lower()
                        if lk in pixels_for_class:
                            pixels_for_class[lk] += px
                    for k, cnt in det_counts.items():
                        lk = k.lower()
                        if lk in counts_for_class:
                            counts_for_class[lk] += cnt
                    overlay = (0.6 * img + 0.4 * masks).astype(np.uint8)
                    coverage = masks.any(axis=2).sum() / self.total_pixels * 100
                    self._update_model_widgets(name, overlay, masks, coverage, pixels_for_class, counts_for_class)

                elif name == 'UNet' and torch is not None:
                    pil = Image.open(self.image_path).convert('RGB')
                    orig_size = pil.size
                    inp = transforms.ToTensor()(pil.resize((256,256))).unsqueeze(0).to(self.device)
                    with torch.no_grad():
                        out = model(inp)
                        mask = torch.argmax(out, dim=1).squeeze(0).cpu().numpy()
                    mask_resized = cv2.resize(mask.astype(np.uint8), orig_size, interpolation=cv2.INTER_NEAREST)
                    masks = np.zeros((orig_size[1], orig_size[0], 3), dtype=np.uint8)
                    # map numeric classes to our canonical names (common repo mapping)
                    idx_to_name = {1: 'animal', 2: 'waste', 3: 'person'}
                    pixels_for_class = {k: 0 for k in CLASS_ORDER}
                    counts_for_class = {k: 0 for k in CLASS_ORDER}
                    for idx, col in {1: CLASS_COLORS['animal'], 2: CLASS_COLORS['waste'], 3: CLASS_COLORS['person']}.items():
                        masks[mask_resized==idx] = col
                        if idx in idx_to_name:
                            cname = idx_to_name[idx]
                            pixels_for_class[cname] = int((mask_resized==idx).sum())
                            # estimate instance counts via connected components for animals/persons
                            if cname in ('animal', 'person'):
                                bin_mask = (mask_resized==idx).astype('uint8')
                                if bin_mask.sum() > 0:
                                    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(bin_mask, connectivity=8)
                                    # subtract background label 0
                                    inst = 0
                                    for s in stats[1:]:
                                        if s[cv2.CC_STAT_AREA] >= 30:
                                            inst += 1
                                    counts_for_class[cname] = inst
                    img = cv2.cvtColor(cv2.imread(self.image_path), cv2.COLOR_BGR2RGB)
                    overlay = (0.6 * img + 0.4 * masks).astype(np.uint8)
                    coverage = (sum(pixels_for_class.values()) / self.total_pixels) * 100 if self.total_pixels>0 else 0.0
                    self._update_model_widgets(name, overlay, masks, coverage, pixels_for_class, counts_for_class)

                else:
                    # Generic torch model: assume it outputs logits with shape (1, C, H, W) or (1, H, W)
                    if torch is None:
                        raise RuntimeError('Torch missing')
                    pil = Image.open(self.image_path).convert('RGB')
                    input_tensor = transforms.Compose([
                        transforms.Resize((256,256)),
                        transforms.ToTensor()
                    ])(pil).unsqueeze(0).to(self.device)
                    with torch.no_grad():
                        out = model(input_tensor)
                        if isinstance(out, (list, tuple)):
                            out = out[0]
                        if out.dim()==4:
                            mask = torch.argmax(out, dim=1).squeeze(0).cpu().numpy()
                        elif out.dim()==3:
                            mask = out.squeeze(0).cpu().numpy()
                        else:
                            mask = np.zeros((input_tensor.shape[2], input_tensor.shape[3]), dtype=np.uint8)
                    mask_resized = cv2.resize(mask.astype(np.uint8), (pil.width, pil.height), interpolation=cv2.INTER_NEAREST)
                    masks = np.zeros((pil.height, pil.width, 3), dtype=np.uint8)
                    # Try to map numeric outputs to canonical classes if possible
                    idx_to_name = {1: 'animal', 2: 'waste', 3: 'person'}
                    pixels_for_class = {k: 0 for k in CLASS_ORDER}
                    counts_for_class = {k: 0 for k in CLASS_ORDER}
                    uniq = np.unique(mask_resized)
                    for v in uniq:
                        if v == 0:
                            continue
                        cname = idx_to_name.get(int(v), None)
                        color = None
                        if cname in CLASS_COLORS:
                            color = CLASS_COLORS[cname]
                        else:
                            color = tuple(np.random.randint(60,255,3).tolist())
                        masks[mask_resized==v] = color
                        if cname:
                            pixels_for_class[cname] = int((mask_resized==v).sum())
                            if cname in ('animal', 'person'):
                                bin_mask = (mask_resized==v).astype('uint8')
                                if bin_mask.sum() > 0:
                                    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(bin_mask, connectivity=8)
                                    inst = 0
                                    for s in stats[1:]:
                                        if s[cv2.CC_STAT_AREA] >= 30:
                                            inst += 1
                                    counts_for_class[cname] = inst
                    img = cv2.cvtColor(cv2.imread(self.image_path), cv2.COLOR_BGR2RGB)
                    overlay = (0.6 * img + 0.4 * masks).astype(np.uint8)
                    coverage = (sum(pixels_for_class.values()) / self.total_pixels) * 100 if self.total_pixels>0 else 0.0
                    self._update_model_widgets(name, overlay, masks, coverage, pixels_for_class, counts_for_class)

            except Exception as e:
                print(f'Error running {name}: {e}')
                self.model_widgets[name]['coverage'].config(text=f'Error: {e}')
        self.status.config(text='All models processed')

    def _update_model_widgets(self, name, overlay, masks, coverage, classes, detections=None):
        # overlay: RGB numpy
        pil_ov = Image.fromarray(overlay)
        pil_ov.thumbnail((900,700), Image.Resampling.LANCZOS)
        imgtk = ImageTk.PhotoImage(pil_ov)
        lbl = self.model_widgets[name]['label']
        lbl.config(image=imgtk, text='')
        lbl.image = imgtk

        # mask thumbnail (centered preview with black background)
        preview = make_mask_preview(masks, out_size=(280,180), bg=(0,0,0))
        masktk = ImageTk.PhotoImage(preview)
        self.model_widgets[name]['mask'].config(image=masktk, text='')
        self.model_widgets[name]['mask'].image = masktk

        self.model_widgets[name]['coverage'].config(text=f'Coverage: {coverage:.2f}%')
        # populate legend / stats area
        stats_frame = self.model_widgets[name].get('stats_frame')
        if stats_frame is not None:
            # clear previous
            for c in stats_frame.winfo_children():
                c.destroy()

            tk.Label(stats_frame, text='Legend / Stats', bg='#f7f9fa', font=('Segoe UI', 9, 'bold')).pack(anchor='w')

            # Build ordered keys: canonical classes first, then extras
            extras = []
            raw_keys = list(classes.keys()) if classes else []
            if detections:
                raw_keys += [k for k in detections.keys() if k not in raw_keys]

            for k in raw_keys:
                lk = k.lower()
                if lk not in CLASS_ORDER and lk not in extras:
                    extras.append(lk)

            ordered_keys = CLASS_ORDER + extras

            # If nothing detected at all
            has_any = any((int(classes.get(k, 0)) if isinstance(classes.get(k,0), (int,float)) else 0) for k in classes) if classes else False
            if not has_any and not extras:
                tk.Label(stats_frame, text='No detections', bg='#f7f9fa').pack(anchor='w')
            else:
                for k in ordered_keys:
                    pixels = int(classes.get(k, 0)) if classes and k in classes else 0
                    det_count = int(detections.get(k, 0)) if detections and k in detections else 0

                    # determine color
                    def _get_color_for(name):
                        ln = name.lower()
                        if ln in CLASS_COLORS:
                            return CLASS_COLORS[ln]
                        if ln.startswith('class_'):
                            try:
                                idx = int(ln.split('_',1)[1])
                                np.random.seed(idx)
                                return tuple(int(x) for x in np.random.randint(80,240,3))
                            except Exception:
                                pass
                        h = abs(hash(ln)) % (2**32)
                        np.random.seed(h)
                        return tuple(int(x) for x in np.random.randint(80,240,3))

                    color = _get_color_for(k)
                    hexc = f"#{color[0]:02x}{color[1]:02x}{color[2]:02x}"

                    row = tk.Frame(stats_frame, bg='#f7f9fa')
                    row.pack(fill=tk.X, pady=2)
                    color_box = tk.Label(row, bg=hexc, width=2, relief=tk.SUNKEN)
                    color_box.pack(side=tk.LEFT, padx=(2,8))

                    title = k.replace('_', ' ').title()
                    info_text = f"{title}"
                    if det_count > 0:
                        info_text += f"  • Count: {det_count}"
                    info_text += f"  • Pixels: {pixels:,}"
                    cov = (pixels / self.total_pixels * 100) if self.total_pixels>0 else 0.0
                    info_text += f"  • {cov:.2f}%"

                    tk.Label(row, text=info_text, bg='#f7f9fa', anchor='w', justify='left').pack(side=tk.LEFT)

        # small log
        print(f'{name} classes:', classes, 'detections:', detections)

# ----------------- Run ----------------- #
if __name__ == '__main__':
    root = tk.Tk()
    app = MultiModelGUI(root)
    root.mainloop()
