"""
图片工具: letterbox resize

保持宽高比, 短边缩放到 target, 长边黑边填充到 target.
YOLO 标签是归一化坐标 (0-1), resize 后完全不需要改动.
"""
from PIL import Image
from typing import Tuple

CANVAS_COLOR = (114, 114, 114)  # YOLO 默认 letterbox 灰


def letterbox_resize(img_path: str, out_path: str, target: Tuple[int, int]) -> Tuple[int, int]:
    """把图片 letterbox 到 target 尺寸, 返回 (原 w, 原 h).

    Args:
        img_path: 输入路径
        out_path: 输出路径 (同格式)
        target: (width, height)
    """
    tw, th = target
    with Image.open(img_path) as img:
        # 保留原格式
        fmt = img.format or "JPEG"
        iw, ih = img.size

        scale = min(tw / iw, th / ih)
        nw, nh = int(iw * scale), int(ih * scale)

        img_resized = img.resize((nw, nh), Image.BILINEAR)

        # 居中贴到灰底画布
        canvas = Image.new("RGB", (tw, th), CANVAS_COLOR)
        pad_left = (tw - nw) // 2
        pad_top = (th - nh) // 2
        canvas.paste(img_resized, (pad_left, pad_top))

        # 保存时根据原格式决定后缀
        if fmt.upper() in ("JPEG", "JPG"):
            canvas.save(out_path, "JPEG", quality=95)
        else:
            canvas.save(out_path, fmt)

    return (iw, ih)


def letterbox_array(img_array, target: Tuple[int, int]):
    """对 numpy BGR 数组做 letterbox, 返回 (resized_array, scale, pad_left, pad_top).

    用于 window_capture 实时 letterbox.
    """
    import numpy as np
    h, w = img_array.shape[:2]
    tw, th = target
    scale = min(tw / w, th / h)
    nw, nh = int(w * scale), int(h * scale)

    img_resized = Image.fromarray(img_array[:, :, ::-1])  # BGR -> RGB
    img_resized = img_resized.resize((nw, nh), Image.BILINEAR)

    canvas = Image.new("RGB", (tw, th), CANVAS_COLOR)
    pad_left = (tw - nw) // 2
    pad_top = (th - nh) // 2
    canvas.paste(img_resized, (pad_left, pad_top))

    return np.array(canvas)[:, :, ::-1].copy(), scale, pad_left, pad_top
