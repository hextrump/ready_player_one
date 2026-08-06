"""
Map ID 手工标注工具 - Web 版

功能:
  - 加载 data/auto_dataset/images/ 里未标注的图片
  - 同时显示全图 + minimap 区域 (左上次区域)
  - 提供 meowdb minimap 缩略图供参考 (按区域分组)
  - 输入 9 位 map ID (e.g. 010001010) 保存到 data/auto_dataset/map_ids.json
  - 快捷键: Enter=保存下一张, ←/→=前后翻, S=跳过

用法:
  python tools/web_map_id.py [--port 8081]
访问: http://localhost:8081
"""
import argparse
import json
import re
from pathlib import Path

import cv2
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS

PROJECT_ROOT = Path(__file__).resolve().parent.parent
IMG_DIR = PROJECT_ROOT / "data" / "auto_dataset" / "images"
MAP_IDS_FILE = IMG_DIR.parent / "map_ids.json"
MINIMAP_DIR = PROJECT_ROOT / "data" / "map_db" / "minimaps"
INDEX_FILE = PROJECT_ROOT / "data" / "map_db" / "index.json"

# minimap 区域 (Artale top-left)
MM_REGION = (5, 80, 260, 260)


def load_index():
    return json.loads(INDEX_FILE.read_text(encoding="utf-8"))


def load_existing_map_ids():
    if MAP_IDS_FILE.exists():
        return json.loads(MAP_IDS_FILE.read_text(encoding="utf-8"))
    return {}


def save_map_ids(d):
    MAP_IDS_FILE.write_text(json.dumps(d, indent=2), encoding="utf-8")


def img_to_b64(img):
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    ok, buf = cv2.imencode(".jpg", img_rgb, [cv2.IMWRITE_JPEG_QUALITY, 85])
    import base64
    return base64.b64encode(buf.tobytes()).decode("ascii") if ok else None


app = Flask(__name__)
CORS(app)


@app.route("/")
def index():
    return send_from_directory(str(Path(__file__).parent), "web_map_id.html")


@app.route("/api/list")
def list_images():
    index = load_index()
    existing = load_existing_map_ids()
    annotated = set(existing.keys())
    images = sorted(IMG_DIR.glob("*.jpg"))
    items = [
        {
            "name": p.name,
            "annotated": p.name in annotated,
            "map_id": existing.get(p.name, ""),
        }
        for p in images
    ]
    return jsonify({
        "total": len(images),
        "annotated": len(annotated),
        "items": items,
        "map_ids": index["map_ids"],
        "minimap_count": index["minimap_downloaded"],
    })


@app.route("/api/image/<path:img_name>")
def get_image(img_name):
    img_path = IMG_DIR / img_name
    if not img_path.exists():
        return jsonify({"error": "not found"}), 404
    img = cv2.imread(str(img_path))
    if img is None:
        return jsonify({"error": "cannot read"}), 400

    # 全图缩到 800 宽预览
    h, w = img.shape[:2]
    full_small = cv2.resize(img, (800, int(h * 800 / w)))

    # minimap 区域单独裁出 (按 1366x768 区域)
    # 如果不是 1366x768, 等比缩放区域
    sx, sy = w / 1366, h / 768
    x1, y1, x2, y2 = MM_REGION
    mm_region = (
        int(x1 * sx), int(y1 * sy),
        int(x2 * sx), int(y2 * sy),
    )
    mm_crop = img[mm_region[1]:mm_region[3], mm_region[0]:mm_region[2]]

    return jsonify({
        "name": img_name,
        "width": w,
        "height": h,
        "full": img_to_b64(full_small),
        "minimap": img_to_b64(mm_crop),
        "region": list(mm_region),
    })


@app.route("/api/minimap/<map_id>")
def get_minimap(map_id):
    """返回 meowdb minimap, 用于参考"""
    p = MINIMAP_DIR / f"{map_id}.png"
    if not p.exists():
        return jsonify({"error": "not found"}), 404
    img = cv2.imread(str(p))
    img = cv2.resize(img, (260, 200))  # 跟游戏 minimap 区域等大
    return jsonify({"image": img_to_b64(img)})


@app.route("/api/save", methods=["POST"])
def save():
    data = request.json or {}
    img_name = data.get("image_name", "")
    map_id = data.get("map_id", "").strip()

    if not re.match(r'^\d{9}$', map_id):
        return jsonify({"error": "map_id 必须 9 位数字"}), 400

    existing = load_existing_map_ids()
    existing[img_name] = map_id
    save_map_ids(existing)
    return jsonify({"status": "ok"})


@app.route("/api/clear", methods=["POST"])
def clear_one():
    data = request.json or {}
    img_name = data.get("image_name", "")
    existing = load_existing_map_ids()
    if img_name in existing:
        del existing[img_name]
        save_map_ids(existing)
    return jsonify({"status": "ok"})


def main():
    p = argparse.ArgumentParser(description="Map ID 手工标注")
    p.add_argument("--port", type=int, default=8081)
    args = p.parse_args()

    if not IMG_DIR.is_dir():
        raise SystemExit(f"未找到 {IMG_DIR}")
    MAP_IDS_FILE.parent.mkdir(parents=True, exist_ok=True)

    print(f"启动: http://localhost:{args.port}")
    print(f"图片: {IMG_DIR}")
    print(f"保存: {MAP_IDS_FILE}")
    app.run(host="0.0.0.0", port=args.port, debug=False)


if __name__ == "__main__":
    main()