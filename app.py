import streamlit as st
from PIL import Image, ImageDraw
from ultralytics import YOLO
import json
import os

# ----------------------------------------------------------
MODEL_PATH = "best.pt"
JSON_LABEL_PATH = "labels.json"   # used only if no .txt exists
TARGET_SIZE = (640, 640)
# ----------------------------------------------------------

@st.cache_resource
def load_model():
    return YOLO(MODEL_PATH)

@st.cache_data
def load_json_annotations(json_path):
    if not os.path.exists(json_path):
        return {}
    with open(json_path, "r") as f:
        data = json.load(f)

    images = {item["id"]: item for item in data.get("images", [])}
    ann_by_image = {}
    for a in data.get("annotations", []):
        ann_by_image.setdefault(a["image_id"], []).append(a)

    gt = {}
    for img_id, anns in ann_by_image.items():
        meta = images[img_id]
        polys = [ann["segmentation"][0] for ann in anns]
        gt[meta["file_name"]] = {
            "orig_width":  meta["width"],
            "orig_height": meta["height"],
            "polygons": polys
        }
    return gt

def draw_polygons(img, polygons, label_text=None, outline="red"):
    draw = ImageDraw.Draw(img)
    for p in polygons:
        pts = [(p[i], p[i+1]) for i in range(0, len(p), 2)]
        draw.polygon(pts, outline=outline)
        if label_text:
            draw.text(pts[0], label_text, fill=outline)
    return img

def read_txt_polygons(txt_path, img_w, img_h):
    """Read YOLO txt polygon file and return pixel coordinates (flat lists)"""
    polys = []
    with open(txt_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            coords = list(map(float, parts[1:]))
            flat = []
            for i in range(0, len(coords), 2):
                x = coords[i]   * img_w
                y = coords[i+1] * img_h
                flat.extend([x, y])
            polys.append(flat)
    return polys

# ----------------------------------------------------------

st.title("HCLTech Vision X: Quality for Bend and Mark")

mode = st.selectbox("Display mode", ["ground-truth", "predictions", "both"])

uploaded = st.file_uploader("Upload an image", type=["jpg", "png"])
if uploaded:
    image = Image.open(uploaded).convert("RGB")
    base_fn = os.path.basename(uploaded.name)

    # resize
    image = image.resize(TARGET_SIZE)
    img_w, img_h = image.size

    summary = {"gt": 0, "pred": 0, "pred_classes": []}

    # ---------- ground-truth polygons ----------
    if mode in ["ground-truth", "both"]:
        txt_path = os.path.splitext(base_fn)[0] + ".txt"
        gt_polys = []

        # 1) Try .txt first
        if os.path.exists(txt_path):
            gt_polys = read_txt_polygons(txt_path, img_w, img_h)

        # 2) Fallback to JSON
        else:
            gt_data = load_json_annotations(JSON_LABEL_PATH)
            if base_fn in gt_data:
                info = gt_data[base_fn]
                sx = TARGET_SIZE[0] / info["orig_width"]
                sy = TARGET_SIZE[1] / info["orig_height"]
                for poly in info["polygons"]:
                    scaled = []
                    for i in range(0, len(poly), 2):
                        scaled.extend([poly[i]*sx, poly[i+1]*sy])
                    gt_polys.append(scaled)

        if gt_polys:
            summary["gt"] = len(gt_polys)
            image = draw_polygons(image, gt_polys, label_text="GT", outline="green")
        else:
            st.warning(f"No ground-truth polygons found for {base_fn}")

    # ---------- model predictions ---------------
    if mode in ["predictions", "both"]:
        model = load_model()
        results = model.predict(image, verbose=False, conf=0.1)
        masks = results[0].masks
        if masks is not None and masks.xy is not None:
            for poly, cid in zip(masks.xy, results[0].boxes.cls.tolist()):
                cls_name = model.names[int(cid)]
                pts = poly.flatten().tolist()
                summary["pred_classes"].append(cls_name)
                image = draw_polygons(image, [pts], label_text=cls_name, outline="red")
            summary["pred"] = len(summary["pred_classes"])
        else:
            st.info("No predictions detected.")

    st.image(image, caption="Result", use_container_width=True)

    with st.expander("Show summary"):
        st.write(f"Ground-truth polygons: {summary['gt']}")
        st.write(f"Predicted polygons :  {summary['pred']}  ({', '.join(summary['pred_classes'])})")

else:
    st.info("📤 Upload an image to begin.")
