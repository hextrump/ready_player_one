"""
诊断 v2: 用真实尺寸验证 抓屏 → 检测。
1) 窗口画 140x140 白方块 (同视频比例 ~3.8% 帧), grab 后 detect 应 active=True
2) 直接喂 01.mp4 原帧 (白图形 0.7~1.2%) 进 detect, 应 active=True

不移动鼠标, 不依赖按键。
"""
import sys
import time
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.capture.window_capture import WindowCapture
from src.perception.lie_detector import LieDetectorModel
from src.utils.config import load_config

WIN = "lie-diag"


def main() -> None:
    ld_cfg = load_config().get("lie_detector", {})
    repo = ld_cfg.get("detector_repo_path") or str(PROJECT_ROOT / "models" / "lie_detector")
    model = LieDetectorModel(repo, backend=ld_cfg.get("backend", "opencv"), config={})

    # ── 1) 窗口: 暗底 + 140x140 中央白方块 (同真实视频比例) ──
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN, 960, 540)
    frame = np.zeros((540, 960, 3), dtype=np.uint8)
    sq = 140
    x0, y0 = (960 - sq) // 2, (540 - sq) // 2
    cv2.rectangle(frame, (x0, y0), (x0 + sq, y0 + sq), (255, 255, 255), -1)
    cv2.imshow(WIN, frame)
    cv2.waitKey(1)

    wc = WindowCapture(process_name="", window_title=WIN)
    for _ in range(50):
        if wc.find_window():
            break
        time.sleep(0.1)
    if not wc.is_valid:
        print("FAIL: 找不到窗口")
        cv2.destroyAllWindows()
        sys.exit(1)
    print(f"窗口已捕获 hwnd={wc.hwnd} 客户区={wc._width}x{wc._height}")

    print("\n── 1) 抓屏窗口(140x140 白方块) → 检测 ──")
    for i in range(6):
        cv2.waitKey(50)
        g = wc.grab()
        if g is None or g.size == 0:
            print(f"grab[{i}] → None/空")
            continue
        res = model.update(g)
        white = int((g > 200).sum() / 3)
        print(f"grab[{i}] {g.shape[1]}x{g.shape[0]} white_px={white} "
              f"active={res.active} conf={res.confidence:.2f} center={res.target_center} "
              f"bbox={res.target_bbox}")
    cv2.destroyAllWindows()

    # ── 2) 直接喂视频原帧 ──
    video = PROJECT_ROOT / "tools" / "01.mp4"
    if video.is_file():
        cap = cv2.VideoCapture(str(video))
        print("\n── 2) 视频原帧 → 检测 (选 4 帧) ──")
        for fi in (0, 24, 72, 144):
            cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
            ok, fr = cap.read()
            if not ok:
                continue
            model.reset()
            for _ in range(2):  # 去抖: 同一帧喂 2 次
                res = model.update(fr)
            white = int((fr > 200).sum() / 3)
            print(f"帧{fi} ({fr.shape[1]}x{fr.shape[0]}) white_px={white} "
                  f"active={res.active} conf={res.confidence:.2f} center={res.target_center} "
                  f"phase={res.phase.value}")
        cap.release()
    else:
        print(f"\n(无视频 {video.name}, 跳过 2)")

    print("诊断结束")


if __name__ == "__main__":
    main()
