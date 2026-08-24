"""批量跑 data/detect/ 下的测谎视频: 每帧 detect → 画框 → 输出带框 mp4 + 每视频统计。

用法:
    python tools/run_detect_videos.py                    # hybrid + UETrack (生产配置)
    python tools/run_detect_videos.py --sot 0            # 纯 hybrid, 无 SOT 对照
    python tools/run_detect_videos.py --remote           # 远程 A/B: 逐帧 POST hhh /frame (全远程)
    python tools/run_detect_videos.py --dir data/detect  # 指定目录

输出: <视频名>_boxed.mp4 (同目录), 统计写 run_summary.txt。
绿色框 = 目标框; 红点 = 融合中心; 左上角黄字 = 帧号/激活/置信/后端状态。
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.perception.lie_detector.hybrid_backend import HybridBackend
from src.perception.lie_detector.remote_backend import RemoteBackend
from src.utils.config import load_config


def draw_overlay(fr, name: str, i: int, r, sot_label: str) -> np.ndarray:
    if r.active and r.target_bbox is not None:
        x1, y1, x2, y2 = [int(v) for v in r.target_bbox]
        cv2.rectangle(fr, (x1, y1), (x2, y2), (0, 255, 0), 2)
        if r.target_center is not None:
            cv2.circle(fr, r.target_center, 4, (0, 0, 255), -1)
    status = "ACT" if r.active else "--"
    cv2.putText(fr, f"{name} f{i} {status} conf={r.confidence:.2f} {sot_label}",
                (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2, cv2.LINE_AA)
    return fr


def _build_remote(ld: dict, host: str | None = None, port: int | None = None) -> RemoteBackend:
    rc = ld.get("remote") or {}
    host = host or str(rc.get("host", "127.0.0.1"))
    port = port if port is not None else int(rc.get("port", 8600))
    print(f"REMOTE 模式: {host}:{port} (需要服务在线; 无则全部 inactive)")
    return RemoteBackend(
        host=host,
        port=port,
        timeout=float(rc.get("timeout", 1.0)),
        jpeg_quality=int(rc.get("jpeg_quality", 85)),
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=str(PROJECT_ROOT / "data" / "detect"))
    ap.add_argument("--sot", type=int, default=1, help="1=UETrack SOT (生产), 0=纯 hybrid 对照; --remote 时忽略")
    ap.add_argument("--remote", action="store_true", help="远程 A/B: 逐帧 POST /frame")
    ap.add_argument("--host", default=None, help="远程服务 host (默认 config.remote.host)")
    ap.add_argument("--port", type=int, default=None, help="远程服务 port (默认 config.remote.port)")
    ap.add_argument("--out-suffix", default="_boxed")
    args = ap.parse_args()

    ld = load_config()["lie_detector"]
    remote = _build_remote(ld, args.host, args.port) if args.remote else None

    hy = None
    hb = None
    if remote is None:
        hy = dict(ld["hybrid"])
        if args.sot == 0:
            hy["uetrack"] = dict(hy["uetrack"])
            hy["uetrack"]["enabled"] = False
        hb = HybridBackend(ld["detector_repo_path"], hy)
        print(f"SOT ready: {hb.sot_ready}")

    # 排除输出文件 (…_boxed.mp4): 否则会把上轮的带框视频再当输入, 叠成 _boxed_boxed
    out_suffix = args.out_suffix
    videos = sorted(
        p for p in Path(args.dir).glob("*.mp4")
        if not p.name.endswith(f"{out_suffix}.mp4")
    )
    if not videos:
        print(f"目录无 mp4: {args.dir}")
        sys.exit(1)

    print(f"视频数: {len(videos)}")

    summary = []
    for v in videos:
        cap = cv2.VideoCapture(str(v))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        W, H = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        out = v.parent / f"{v.stem}{args.out_suffix}.mp4"
        writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))

        if hb is not None:
            hb.reset()   # 每视频重来 bg/kalman/SOT 模板; UETrack 模型保持已构建
        if remote is not None:
            remote.clear()   # 重置服务端去抖/会话, 每视频公平对比

        n = active = 0
        confs, lat = [], []
        sot_inited_ever = False
        while True:
            ok, fr = cap.read()
            if not ok:
                break
            n += 1
            t0 = time.perf_counter()
            if remote is not None:
                r = remote.update(fr)
                label = "remote"
            else:
                r = hb.detect(fr)
                if hb.sot_inited:
                    sot_inited_ever = True
                label = "SOT" if hb.sot_inited else "nosot"
            lat.append(time.perf_counter() - t0)
            if r.active:
                active += 1
                confs.append(r.confidence)
            writer.write(draw_overlay(fr, v.stem, n, r, label))
        cap.release()
        writer.release()

        ms = float(np.mean(lat)) * 1000 if lat else 0.0
        conf = float(np.mean(confs)) if confs else 0.0
        line = (f"{v.name}: {n}帧 active {active}/{n} ({100 * active / max(n, 1):.0f}%) "
                f"conf均值 {conf:.2f} sot={sot_inited_ever} 帧延迟 {ms:.0f}ms -> {out.name}")
        print(line)
        summary.append(line)

    Path(args.dir, "run_summary.txt").write_text("\n".join(summary), encoding="utf-8")


if __name__ == "__main__":
    main()
