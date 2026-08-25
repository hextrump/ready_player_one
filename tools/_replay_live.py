"""
逐帧回放诊断 — 把测谎视频每帧发给真实 hhh 服务端, 记录响应流
================================================================
目的: 定位 opencv/samurai "跟歪" 的具体模式。
每帧 POST /frame → {active, phase, confidence, center, s_bbox}。
从响应流推断:
  - countdown 阶段 center 跳变 = opencv 错选 (白块多选/星形部分命中)
  - tracking 阶段 s_bbox 缺失 = 锚点守卫触发 (opencv 拽回) 或 samurai step 失败
  - conf 持续走低 = 检测弱

用法:
  python tools/_replay_live.py <video> [--out <json>] [--fps N] [--host H] [--port P]

保真度 (重要): 生产抓帧是 WindowCapture 默认 letterbox 到 CANONICAL_SIZE (1366,768),
服务端看到的是 1366x768 画布帧, 距离门/尺寸门都是绝对 px。直接 POST 原生视频帧会让
replay 与生产几何不一致 (1280x720 好视频在 replay 星形比生产大 1.33x, 744x480 坏视频
反而小), 这是"replay 全 PASS、实机仍歪"的最可能原因。本工具默认把每帧
letterbox_array(frame, (1366,768)) 后再 POST, 逐帧复刻生产几何。
"""
from __future__ import annotations

import argparse
import base64
import http.client
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.image_utils import letterbox_array  # noqa: E402  生产同款 letterbox

# 生产 WindowCapture CANONICAL_SIZE (src/capture/window_capture.py)
CANVAS_W, CANVAS_H = 1366, 768


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--out", default="")
    ap.add_argument("--fps", type=int, default=5,
                    help="喂帧节奏 (默认 5, 匹配生产视觉线程 3-5fps; 旧默认 30 会让 miss/稳定计数失真)")
    ap.add_argument("--host", default="100.118.47.94")
    ap.add_argument("--port", type=int, default=8600)
    ap.add_argument("--max-frames", type=int, default=0)
    ap.add_argument("--canvas-w", type=int, default=CANVAS_W)
    ap.add_argument("--canvas-h", type=int, default=CANVAS_H)
    args = ap.parse_args()

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"打不开视频: {args.video}")
        return 1

    conn = http.client.HTTPConnection(args.host, args.port, timeout=10)
    rows = []
    idx = 0
    t_prev = None
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if args.max_frames and idx >= args.max_frames:
            break
        # 保真: 复刻生产 WindowCapture 几何 — 原生帧 letterbox 到 (1366,768) 灰底画布
        frame, _scale, _pad_l, _pad_t = letterbox_array(frame, (args.canvas_w, args.canvas_h))
        # 节奏: 模拟实时喂帧
        now = time.time()
        if t_prev is not None and args.fps > 0:
            dt = 1.0 / args.fps
            sl = dt - (now - t_prev)
            if sl > 0:
                time.sleep(sl)
        t_prev = time.time()

        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        body = json.dumps({"image_b64": base64.b64encode(buf.tobytes()).decode("ascii")}).encode()
        try:
            conn.request("POST", "/frame", body, {"Content-Type": "application/json"})
            resp = conn.getresponse()
            data = json.loads(resp.read())
        except Exception as e:
            rows.append({"idx": idx, "error": f"{type(e).__name__}: {e}"})
            idx += 1
            continue
        rows.append({
            "idx": idx,
            "t": round(time.time() - t_start, 3),
            "active": data.get("active"),
            "phase": data.get("phase"),
            "conf": data.get("confidence"),
            "center": data.get("center"),
            "bbox": data.get("bbox"),
            "s_bbox": data.get("s_bbox"),
            "diag": data.get("diag"),
        })
        idx += 1
    cap.release()
    conn.close()

    if args.out:
        Path(args.out).write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")
    # 摘要
    analyze(rows)
    return 0


def analyze(rows) -> None:
    n = len(rows)
    act = [r for r in rows if r.get("active")]
    nact = len(act)
    print(f"帧={n}  active帧={nact} ({100*nact/max(1,n):.0f}%)")

    # phase 分布 (active 内)
    from collections import Counter
    ph = Counter(r.get("phase") for r in act)
    print(f"active phase: {dict(ph)}")

    # countdown 阶段 center 跳变 (opencv 错选指标): 相邻 active 同 phase 帧 center 位移
    jumps = []
    prev = None
    for r in act:
        c = r.get("center")
        if c and r.get("phase") == "countdown" and prev and prev[0] == "countdown":
            d = np.hypot(c[0] - prev[1][0], c[1] - prev[1][1])
            if d > 20:  # 星形 ~31x23, 中心跳>20px 可疑
                jumps.append((r["idx"], round(float(d))))
        prev = (r.get("phase"), c) if c else None
    if jumps:
        print(f"countdown 中心跳>20px: {len(jumps)}次 最大={max(j for _, j in jumps)}px")
        print("  前10:", jumps[:10])
    else:
        print("countdown 中心跳>20px: 0次 (opencv 选择稳定)")

    # tracking 阶段 s_bbox 缺失率 + conf
    tr = [r for r in act if r.get("phase") == "tracking"]
    if tr:
        no_bbox = [r for r in tr if not r.get("s_bbox")]
        confs = [r.get("conf") or 0 for r in tr]
        print(f"tracking帧={len(tr)}  s_bbox缺失={len(no_bbox)} ({100*len(no_bbox)/len(tr):.0f}%)  "
              f"conf中位={sorted(confs)[len(confs)//2]:.2f}")
        if no_bbox:
            print("  缺失样例 idx:", [r["idx"] for r in no_bbox][:15])

    # tracking 阶段 center 跳变 (samurai 漂移/守卫拽回指标)
    tj = []
    prev = None
    for r in tr:
        c = r.get("center")
        if c and prev and prev[0] == "tracking":
            d = np.hypot(c[0] - prev[1][0], c[1] - prev[1][1])
            if d > 60:
                tj.append((r["idx"], round(float(d))))
        prev = (r.get("phase"), c) if c else None
    if tj:
        print(f"tracking 中心跳>60px: {len(tj)}次 最大={max(j for _, j in tj)}px")
        print("  前10:", tj[:10])

    # ---- diag 分类: 每个被接受 (>120px) teleport 按 diag 指纹归类 ----
    teleports = []
    prev_c = None
    prev_area = None
    for r in act:
        c = r.get("center")
        d = r.get("diag") or {}
        if not c:
            prev_c = None
            prev_area = None
            continue
        tags = []
        if prev_c is not None:
            dist = float(np.hypot(c[0] - prev_c[0], c[1] - prev_c[1]))
            if dist > 120:
                if d.get("new_event"):
                    tags.append("new_event")
                if d.get("anchor_guard"):
                    tags.append("anchor_guard")
                if (d.get("miss_run") or 0) < 2 and not d.get("new_event"):
                    tags.append("mid_event")
                # 尺寸异常: 被接受 bbox 面积 vs 上一被接受面积
                bbox = d.get("ob") or r.get("bbox")
                if bbox and prev_area:
                    area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
                    if prev_area > 0 and area / prev_area > 4:
                        tags.append(f"size_x{area / prev_area:.0f}")
                teleports.append({
                    "idx": r["idx"], "dist": round(dist), "tags": ",".join(tags) or "-",
                    "branch": d.get("branch"), "phase_raw": d.get("phase_raw"),
                    "conf": round(float(r.get("conf") or 0), 2),
                    "sam_conf": d.get("sam_conf"),
                })
        if c:
            prev_c = c
            bbox = d.get("ob") or r.get("bbox")
            if bbox:
                prev_area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
    if teleports:
        print(f"\n=== 被接受 teleport (>120px) x{len(teleports)} ===")
        for t in teleports:
            print(f"  idx={t['idx']} dist={t['dist']}px {t['tags']}  branch={t['branch']} "
                  f"phase_raw={t['phase_raw']} conf={t['conf']} sam_conf={t['sam_conf']}")
    else:
        print("\n被接受 teleport (>120px): 0次")

    # ---- bbox 面积分布 (校准 SIZE_GATE_FACTOR: 星形 vs 数字) ----
    areas = []
    for r in act:
        d = r.get("diag") or {}
        bbox = d.get("ob") or r.get("bbox")
        if bbox:
            areas.append((bbox[2] - bbox[0]) * (bbox[3] - bbox[1]))
    if areas:
        s = sorted(areas)
        n2 = len(s)
        q = lambda p: s[min(n2 - 1, int(n2 * p))]
        print(f"opencv bbox 面积分布 (accepted帧, n={n2}): p10={q(0.1):.0f} "
              f"p50={q(0.5):.0f} p90={q(0.9):.0f} max={s[-1]:.0f}")


if __name__ == "__main__":
    t_start = time.time()
    sys.exit(main())
