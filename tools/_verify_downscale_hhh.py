"""
hhh 验证: 降采样检测 vs 全帧检测 — 真实视频帧上确认:
  1) 星形目标在 DETECT_MAX_SIDE=320 下仍被检出 (active=True)
  2) 降采样缩回坐标与全帧检测中心一致
用法 (在 hhh 项目目录下):
  python tools/_verify_downscale_hhh.py <frames_dir> <repo>
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

DETECT_MAX_SIDE = 320  # 必须与服务端常量一致


def main() -> int:
    frames_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("tools/_frames")
    repo = sys.argv[2] if len(sys.argv) > 2 else "C:/Users/heyas/Documents/code/lie-detector"

    sys.path.insert(0, str(PROJECT_ROOT / "src"))
    from src.perception.lie_detector.opencv_backend import OpenCVBackend

    ob = OpenCVBackend(repo)
    if not ob.ready:
        print(f"FAIL opencv 不可用: {ob.import_error}")
        return 1

    n_ok = 0
    n_active_full = 0
    n_active_small = 0
    for p in sorted(frames_dir.glob("*.png")):
        bgr = cv2.imread(str(p))
        if bgr is None:
            continue
        H, W = bgr.shape[:2]
        max_side = max(H, W)

        # 全帧检测 (旧路径基线)
        r_full = ob.detect(bgr, scale=1.0)

        # 服务端降采样路径
        scale = 1.0
        small = bgr
        if max_side > DETECT_MAX_SIDE:
            scale = DETECT_MAX_SIDE / max_side
            small = cv2.resize(
                bgr,
                (max(1, int(round(W * scale))), max(1, int(round(H * scale)))),
                interpolation=cv2.INTER_AREA,
            )
        r_small = ob.detect(small, scale=scale)

        line = f"{p.name}: full={r_full.active} conf={r_full.confidence:.2f} ctr={r_full.target_center} | small={r_small.active} conf={r_small.confidence:.2f} ctr={r_small.target_center}"
        if r_full.active and r_small.active:
            dc = np.hypot(
                r_full.target_center[0] - r_small.target_center[0],
                r_full.target_center[1] - r_small.target_center[1],
            )
            line += f" | center_diff={dc:.1f}px"
            n_active_full += 1
            n_active_small += 1
            n_ok += 1 if dc < 15 else 0
        elif r_full.active and not r_small.active:
            line += "   <<< 降采样丢失目标"
            n_active_full += 1
        elif not r_full.active and r_small.active:
            line += "   <<< 全帧未检出但降采样检出"
            n_active_small += 1
        print(line)

    print(f"\n结果: {n_ok} 帧降采样缩回一致 / {n_active_full} 帧全帧检出, {n_active_small} 帧降采样检出")
    return 0 if n_active_full == 0 or n_ok == n_active_full else 1


if __name__ == "__main__":
    sys.exit(main())
