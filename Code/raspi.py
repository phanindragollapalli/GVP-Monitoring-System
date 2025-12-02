'''# app.py
from flask import Flask, render_template_string, request, send_file
from ultralytics import YOLO
import cv2
import numpy as np
import time
import os
import traceback
import logging

# ---------------- config ----------------
logging.basicConfig(filename="error.log", level=logging.ERROR,
                    format="%(asctime)s %(levelname)s %(message)s")

MODEL_PATH = "yolov11.pt"
UPLOAD_DIR = "uploads"
OUT_DIR = "static"
OUT_IMG = os.path.join(OUT_DIR, "segmented.png")

# Class names must match model label order
class_names = {0: "animal", 1: "waste", 2: "person"}

# BGR colors for OpenCV (waste -> bright red)
class_colors = {
    "animal": (0, 255, 0),   # Green
    "waste":  (0, 0, 255),   # Bright Red
    "person": (255, 255, 0)  # Cyan/Yellowish
}

# Simple HTML template (inline)
HTML = """
<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>YOLOv11 Segmentation</title>
  <style>
    body { font-family: Arial, sans-serif; background:#111; color:#eee; text-align:center; padding:30px; }
    .card { background:#1c1c1c; padding:20px; border-radius:8px; display:inline-block; width:85%; }
    input[type=file] { margin-top:12px; }
    img { margin-top:16px; max-width:100%; border-radius:6px; }
    table { margin:16px auto; border-collapse:collapse; color:#eee; }
    th, td { padding:8px 12px; border:1px solid #333; }
    .small { font-size:0.9em; color:#bbb; }
  </style>
</head>
<body>
  <div class="card">
    <h1>Garbage Vulnerability □^`^t Segmentation</h1>
    <form action="/upload" method="post" enctype="multipart/form-data">
      <input type="file" name="image" accept="image/*" required>
      <button type="submit">Segment Image</button>
    </form>

    {% if error %}
      <p style="color: #ff6666;">Error: {{ error }}</p>
    {% endif %}

    {% if segmented %}
      <h2>Result</h2>
      <img src="{{ url_for('display_image', filename='segmented.png') }}" alt="result">
      <h3>Stats</h3>
      <table>
        <tr><th>Class</th><th>Pixels</th><th>Coverage (%)</th></tr>
        {% for row in stats %}
          <tr><td>{{ row[0] }}</td><td>{{ row[1] }}</td><td>{{ "%.3f"|format(row[2]) }}</td></tr>
        {% endfor %}
      </table>
      <p class="small"><b>Inference time:</b> {{ infer_time }} s &nbsp; | &nbsp; <b>Device:</b> CPU (Ultralytics on Pi)</p>
    {% endif %}

    <p class="small">Check <code>error.log</code> for details on any failures.</p>
  </div>
</body>
</html>
"""

# ---------------- startup checks ----------------
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model not found at {MODEL_PATH}. Place yolov11.pt in the working folder.")

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)

# Load model once (may be slow)
model = YOLO(MODEL_PATH)

app = Flask(__name__)

# ---------------- helper functions ----------------
def draw_legend(img, class_names_map, class_colors_map, start_x=10, start_y=10):
    x, y = start_x, start_y
    box_w, box_h = 28, 18
    pad = 8
    font = cv2.FONT_HERSHEY_SIMPLEX
    for k in sorted(class_names_map.keys()):
        name = class_names_map[k]
        color = tuple(map(int, class_colors_map.get(name, (255,255,255))))
        cv2.rectangle(img, (x, y), (x+box_w, y+box_h), color, -1)
        cv2.putText(img, name, (x+box_w+8, y+box_h-3), font, 0.45, (255,255,255), 1, cv2.LINE_AA)
        y += box_h + pad

def safe_resize_mask_to_image(mask, W, H):
    """
    mask: 2D numpy (h, w) or probability map
    returns boolean mask (H, W)
    """
    # convert to float32 then resize with nearest neighbor
    try:
        arr = mask.astype(np.float32)
    except Exception:
        arr = np.array(mask, dtype=np.float32)
    resized = cv2.resize(arr, (W, H), interpolation=cv2.INTER_NEAREST)
    # threshold
    return resized > 0.5

# ---------------- main processing ----------------
def process_and_annotate(img_path):
    """
    Returns: (annotated_bgr_image, stats_list, elapsed_seconds)
     - stats_list: [(class_name, pixel_count, coverage_percent), ...] for all classes
    """
    img = cv2.imread(img_path)
    if img is None:
        raise RuntimeError("Failed to read uploaded image.")
  H, W = img.shape[:2]
    total_pixels = H * W

    # run model (timed)
    t0 = time.time()
    # using model.predict gives the verbose timing; model(img_path) is shorthand. Use predict for explicitness:
    results = model.predict(source=img_path, task="segment", save=False)
    t1 = time.time()
    elapsed = t1 - t0

    # prepare mask image
    mask_img = np.zeros_like(img, dtype=np.uint8)
    pixel_counts = {name: 0 for name in class_names.values()}

    # iterate results (Ultralytics may return list of results)
    for r in results:
        # try to robustly extract masks and class ids
        try:
            if getattr(r, "masks", None) is None:
                continue
            # r.masks.data is common (torch tensor)
            masks = r.masks.data.cpu().numpy()  # shape (N, mh, mw)
        except Exception:
            # fallback: sometimes r.masks might already be numpy-like
            try:
                masks = np.array(r.masks)
            except Exception:
                masks = np.empty((0, H, W))

        try:
            cls_ids = r.boxes.cls.cpu().numpy().astype(int)
        except Exception:
            # fallback: if boxes/cls not available, assume zeros
            cls_ids = np.zeros((masks.shape[0],), dtype=int) if masks.size else np.array([])

        # normalize masks to have shape (N, h, w)
        if masks.ndim == 2:
            masks = masks[np.newaxis, ...]
        elif masks.ndim == 1:
            # try to expand if each item is an array-like
   masks = np.stack(masks) if len(masks) else np.empty((0, H, W))

        # process each mask
        for i, mask in enumerate(masks):
            cls = int(cls_ids[i]) if i < len(cls_ids) else 0
            cls_name = class_names.get(cls, f"class_{cls}")
            color = class_colors.get(cls_name, (0,255,255))  # fallback color

            # resize mask robustly to image resolution
            mask_bool = safe_resize_mask_to_image(mask, W, H)  # boolean (H,W)

            # apply color where mask is True
            color_arr = np.array(color, dtype=np.uint8)
            # mask_img[mask_bool] -> sets pixels for each True location; broadcasting works
            mask_img[mask_bool] = color_arr

            # accumulate counts
            pixel_counts[cls_name] = pixel_counts.get(cls_name, 0) + int(np.sum(mask_bool))

    # blend overlay
    alpha_img = 0.55  # original image weight
    alpha_mask = 0.45
    blended = cv2.addWeighted(img, alpha_img, mask_img, alpha_mask, 0)

    # draw legend
    draw_legend(blended, class_names, class_colors, start_x=12, start_y=12)

    # build stats list (include zero entries too)
    stats = []
    for k in sorted(class_names.keys()):
        name = class_names[k]
        count = pixel_counts.get(name, 0)
        perc = (count / total_pixels * 100) if total_pixels > 0 else 0.0
        stats.append((name, int(count), float(perc)))

    return blended, stats, round(elapsed, 3)

# ---------------- flask routes ----------------
@app.route("/", methods=["GET"])
def index():
   return render_template_string(HTML)

@app.route("/upload", methods=["POST"])
def upload():
    try:
        if "image" not in request.files:
            return render_template_string(HTML, error="No file uploaded")
        f = request.files["image"]
        if f.filename == "":
            return render_template_string(HTML, error="No file selected")
        # save upload
        fname = os.path.basename(f.filename)
        save_path = os.path.join(UPLOAD_DIR, fname)
        f.save(save_path)

        # process
        annotated, stats, elapsed = process_and_annotate(save_path)

        # write result
        cv2.imwrite(OUT_IMG, annotated)

        # return page
        return render_template_string(HTML, segmented=True, stats=stats, infer_time=elapsed)
    except Exception as e:
        tb = traceback.format_exc()
        logging.error(tb)
        return render_template_string(HTML, error="Internal error: check error.log"), 500

@app.route("/display/<filename>")
def display_image(filename):
    # serve static result
    path = os.path.join(OUT_DIR, filename)
    if not os.path.exists(path):
        return "Not found", 404
    return send_file(path, mimetype="image/png")

# ---------------- main ----------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000) '''
# app.py
from flask import Flask, render_template_string, request, send_file
from ultralytics import YOLO
import cv2
import numpy as np
import time
import os
import traceback
import logging

# ---------------- config ----------------
logging.basicConfig(filename="error.log", level=logging.ERROR,
                    format="%(asctime)s %(levelname)s %(message)s")

MODEL_PATH = "yolov11.pt"
UPLOAD_DIR = "uploads"
OUT_DIR = "static"
OUT_IMG = os.path.join(OUT_DIR, "segmented.png")

class_names = {0: "animal", 1: "waste", 2: "person"}

# waste = bright red
class_colors = {
    "animal": (0, 255, 0),
    "waste":  (0, 0, 255),
    "person": (255, 255, 0)
}

HTML = """
<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>YOLOv11 Segmentation</title>
  <style>
    body { font-family: Arial, sans-serif; background:#111; color:#eee; text-align:center; padding:30px; }
    .card { background:#1c1c1c; padding:20px; border-radius:8px; display:inline-block; width:85%; }
    input[type=file] { margin-top:12px; }
   img { margin-top:16px; max-width:100%; border-radius:6px; }
    table { margin:16px auto; border-collapse:collapse; color:#eee; }
    th, td { padding:8px 12px; border:1px solid #333; }
    .small { font-size:0.9em; color:#bbb; }
  </style>
</head>
<body>
  <div class="card">
    <h1>Garbage Vulnerability □^`^t Segmentation</h1>
    <form action="/upload" method="post" enctype="multipart/form-data">
      <input type="file" name="image" accept="image/*" required>
      <button type="submit">Segment Image</button>
    </form>

    {% if error %}
      <p style="color: #ff6666;">Error: {{ error }}</p>
    {% endif %}

    {% if segmented %}
      <h2>Result</h2>
      <img src="{{ url_for('display_image', filename='segmented.png') }}" alt="result">
      <h3>Class Statistics</h3>
      <table>
        <tr><th>Class</th><th>Pixels</th><th>Coverage (%)</th></tr>
        {% for row in stats %}
          <tr><td>{{ row[0] }}</td><td>{{ row[1] }}</td><td>{{ "%.3f"|format(row[2]) }}</td></tr>
        {% endfor %}
      </table>
      <h3>Timing</h3>
      <table>
        <tr><th>Stage</th><th>Wall Time (s)</th><th>CPU Time (s)</th></tr>
        <tr><td>Preprocess</td><td>{{ timings.prep_wall }}</td><td>{{ timings.prep_cpu }}</td></tr>
        <tr><td>Inference</td><td>{{ timings.inf_wall }}</td><td>{{ timings.inf_cpu }}</td></tr>
        <tr><td>Postprocess</td><td>{{ timings.post_wall }}</td><td>{{ timings.post_cpu }}</td></tr>
        <tr><td><b>Total</b></td><td><b>{{ timings.total_wall }}</b></td><td><b>{{ timings.total_cpu }}</b></td></tr>
      </table>
      <p class="small">Device: CPU (Raspberry Pi)</p>
    {% endif %}
    <p class="small">Check <code>error.log</code> for details on any failures.</p>
  </div>
</body>
</html>
"""

# ---------------- startup ----------------
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model not found at {MODEL_PATH}. Place yolov11.pt in the working folder.")

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)

model = YOLO(MODEL_PATH)
app = Flask(__name__)

# ---------------- helpers ----------------
def draw_legend(img, class_names_map, class_colors_map, start_x=10, start_y=10):
    x, y = start_x, start_y
    box_w, box_h = 33, 22
    pad = 15
    font = cv2.FONT_HERSHEY_SIMPLEX
    for k in sorted(class_names_map.keys()):
        name = class_names_map[k]
        color = tuple(map(int, class_colors_map.get(name, (255,255,255))))
        cv2.rectangle(img, (x, y), (x+box_w, y+box_h), color, -1)
        cv2.putText(img, name, (x+box_w+10, y+box_h-5), font, 1.8, (255,255,255), 2, cv2.LINE_AA)
        y += box_h + pad

def safe_resize_mask_to_image(mask, W, H):
    arr = np.array(mask, dtype=np.float32)
    resized = cv2.resize(arr, (W, H), interpolation=cv2.INTER_NEAREST)
    return resized > 0.5
# ---------------- processing ----------------
def process_and_annotate(img_path):

    img = cv2.imread(img_path)
    if img is None:
        raise RuntimeError("Failed to read uploaded image.")
    H, W = img.shape[:2]
    total_pixels = H * W

    # ---- timers ----
    wall_start_total = time.time()
    cpu_start_total  = time.process_time()

    # preprocess (none for now)

    wall_start_prep = time.time()
    cpu_start_prep  = time.process_time()
    wall_end_prep = time.time()
    cpu_end_prep  = time.process_time()

    # inference
    wall_start_inf = time.time()
    cpu_start_inf  = time.process_time()
    results = model.predict(source=img_path, task="segment", save=False)
    wall_end_inf = time.time()
    cpu_end_inf  = time.process_time()

    # postprocess
    wall_start_post = time.time()
    cpu_start_post  = time.process_time()

    mask_img = np.zeros_like(img, dtype=np.uint8)
    pixel_counts = {name: 0 for name in class_names.values()}

    for r in results:
        if getattr(r, "masks", None) is None:
            continue
        masks = r.masks.data.cpu().numpy()
        cls_ids = r.boxes.cls.cpu().numpy().astype(int)

        for i, mask in enumerate(masks):

            cls = int(cls_ids[i]) if i < len(cls_ids) else 0
            cls_name = class_names.get(cls, f"class_{cls}")
            color = class_colors.get(cls_name, (0,255,255))

            mask_bool = safe_resize_mask_to_image(mask, W, H)
            mask_img[mask_bool] = np.array(color, dtype=np.uint8)
            pixel_counts[cls_name] += int(np.sum(mask_bool))

    blended = cv2.addWeighted(img, 0.55, mask_img, 0.45, 0)
    draw_legend(blended, class_names, class_colors, start_x=12, start_y=12)

    wall_end_post = time.time()
    cpu_end_post  = time.process_time()

    wall_end_total = time.time()
    cpu_end_total  = time.process_time()

    # timings summary
    timings = {
        "prep_wall": round(wall_end_prep - wall_start_prep, 3),
        "prep_cpu": round(cpu_end_prep - cpu_start_prep, 3),
        "inf_wall": round(wall_end_inf - wall_start_inf, 3),
        "inf_cpu": round(cpu_end_inf - cpu_start_inf, 3),
        "post_wall": round(wall_end_post - wall_start_post, 3),
        "post_cpu": round(cpu_end_post - cpu_start_post, 3),
        "total_wall": round(wall_end_total - wall_start_total, 3),
        "total_cpu": round(cpu_end_total - cpu_start_total, 3)
    }

    stats = []
    for k in sorted(class_names.keys()):
        name = class_names[k]
        count = pixel_counts.get(name, 0)
        perc = (count / total_pixels * 100) if total_pixels > 0 else 0.0
        stats.append((name, int(count), float(perc)))

# ---------------- routes ----------------
@app.route("/", methods=["GET"])
def index():
    return render_template_string(HTML)

@app.route("/upload", methods=["POST"])
def upload():
    try:
        if "image" not in request.files:
            return render_template_string(HTML, error="No file uploaded")
        f = request.files["image"]
        if f.filename == "":
            return render_template_string(HTML, error="No file selected")

        fname = os.path.basename(f.filename)
        save_path = os.path.join(UPLOAD_DIR, fname)
        f.save(save_path)

        annotated, stats, timings = process_and_annotate(save_path)
        cv2.imwrite(OUT_IMG, annotated)

        return render_template_string(HTML, segmented=True, stats=stats, timings=timings)
    except Exception:
        logging.error(traceback.format_exc())
        return render_template_string(HTML, error="Internal error: check error.log"), 500

@app.route("/display/<filename>")
def display_image(filename):
    path = os.path.join(OUT_DIR, filename)
    if not os.path.exists(path):
        return "Not found", 404
    return send_file(path, mimetype="image/png")

# ---------------- main ----------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
this is mu code, why is my preprocessing time 0?