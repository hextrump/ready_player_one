"""
Web 标注工具 - 标注 data/auto_dataset/

类别 ID (V13): 0=Player, 1=Monster, 2=Platform, 3=Rope

用法:
  python tools/web_annotator.py [--port 8080]
浏览器访问: http://localhost:8080
"""
import argparse
import base64
from pathlib import Path

import cv2
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS

PROJECT_ROOT = Path(__file__).resolve().parent.parent
IMG_DIR = PROJECT_ROOT / "data" / "auto_dataset" / "images"
LABEL_DIR = PROJECT_ROOT / "data" / "auto_dataset" / "labels"

# V13 类别 (连续 0-3)
CLASS_COLORS = {
    0: (0, 136, 255),     # Player - 蓝
    1: (68, 68, 255),     # Monster - 红
    2: (0, 255, 0),       # Platform - 绿
    3: (255, 255, 0),     # Rope - 青
}
CLASS_DISPLAY = {
    0: "Player",
    1: "Monster",
    2: "Platform",
    3: "Rope",
}

app = Flask(__name__)
CORS(app)
LABEL_DIR.mkdir(parents=True, exist_ok=True)


def load_existing(img_name):
    lbl = LABEL_DIR / (Path(img_name).stem + ".txt")
    boxes = []
    if lbl.exists():
        with open(lbl, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) != 5:
                    continue
                try:
                    boxes.append({
                        "cls": int(parts[0]),
                        "cx": float(parts[1]),
                        "cy": float(parts[2]),
                        "bw": float(parts[3]),
                        "bh": float(parts[4]),
                    })
                except ValueError:
                    continue
    return boxes


def save_labels(img_name, boxes):
    lbl = LABEL_DIR / (Path(img_name).stem + ".txt")
    lines = []
    for b in boxes:
        cls = int(b["cls"])
        if cls not in CLASS_DISPLAY:
            continue
        lines.append(
            f"{cls} {float(b['cx']):.6f} {float(b['cy']):.6f} "
            f"{float(b['bw']):.6f} {float(b['bh']):.6f}"
        )
    with open(lbl, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def img_to_base64(img):
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    ok, buf = cv2.imencode(".jpg", img_rgb, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return base64.b64encode(buf.tobytes()).decode("ascii") if ok else None


@app.route("/")
def index():
    return send_from_directory(str(Path(__file__).parent), "web_annotator.html")


@app.route("/api/classes")
def list_classes():
    return jsonify([
        {"id": cid, "name": CLASS_DISPLAY[cid], "color": list(CLASS_COLORS[cid])}
        for cid in CLASS_DISPLAY
    ])


@app.route("/api/images")
def list_images():
    return jsonify(sorted(p.name for p in IMG_DIR.glob("*.jpg")))


@app.route("/api/image/<path:img_name>")
def get_image(img_name):
    img_path = IMG_DIR / img_name
    if not img_path.exists():
        return jsonify({"error": "not found"}), 404
    img = cv2.imread(str(img_path))
    if img is None:
        return jsonify({"error": "cannot read"}), 400

    h, w = img.shape[:2]
    existing = load_existing(img_name)

    preview = img.copy()
    for b in existing:
        if b["cls"] not in CLASS_COLORS:
            continue
        x1 = int((b["cx"] - b["bw"] / 2) * w)
        y1 = int((b["cy"] - b["bh"] / 2) * h)
        x2 = int((b["cx"] + b["bw"] / 2) * w)
        y2 = int((b["cy"] + b["bh"] / 2) * h)
        color = CLASS_COLORS[b["cls"]]
        cv2.rectangle(preview, (x1, y1), (x2, y2), color, 2)
        cv2.putText(preview, CLASS_DISPLAY.get(b["cls"], str(b["cls"])),
                    (x1, max(y1 - 5, 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    return jsonify({
        "name": img_name,
        "width": w,
        "height": h,
        "image": img_to_base64(preview),
        "existing": existing,
    })


@app.route("/api/save", methods=["POST"])
def save_labels_api():
    data = request.json or {}
    img_name = data.get("image_name")
    boxes = data.get("boxes", [])
    if not img_name:
        return jsonify({"error": "no image name"}), 400
    save_labels(img_name, boxes)
    return jsonify({"status": "ok", "saved": len(boxes)})


def main():
    parser = argparse.ArgumentParser(description="Web 标注工具")
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args()
    print(f"启动 Web 标注器: http://localhost:{args.port}")
    print(f"图片目录: {IMG_DIR}")
    print(f"标签目录: {LABEL_DIR}")
    app.run(host="0.0.0.0", port=args.port, debug=False)


if __name__ == "__main__":
    main()
