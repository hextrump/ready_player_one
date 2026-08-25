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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--out", default="")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--host", default="100.118.47.94")
    ap.add_argument("--port", type=int, default=8600)
    ap.add_argument("--max-frames", type=int, default=0)
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


if __name__ == "__main__":
    t_start = time.time()
    sys.exit(main())
