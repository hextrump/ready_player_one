import os
import cv2
import time
from typing import Optional
from src.utils.logger import get_logger

log = get_logger("data_collector")

class DataCollector:
    """
    自动数据收集器 (Active Learning 模块)
    作用：以一定频率，或者在遇到困难样本（如测谎仪 UI、低置信度检测）时，
    把游戏截图和对应的 YOLO 标签（txt）自动按照数据集格式保存下来。
    后续可以直接用这些数据进行模型微调（Fine-tuning）。
    """
    def __init__(self, save_dir: str = "data/auto_dataset"):
        # 严格遵守 YOLO 数据集目录格式
        self.img_dir = os.path.join(save_dir, "images")
        self.lbl_dir = os.path.join(save_dir, "labels")
        self._ensure_dirs()
        
        self.last_save_time = time.time()
        self.save_interval_seconds = 60.0  # 默认 60 秒保存一次常规样本

    def _ensure_dirs(self):
        os.makedirs(self.img_dir, exist_ok=True)
        os.makedirs(self.lbl_dir, exist_ok=True)

    def save_snapshot(self, frame, yolo_results, is_hard_example: bool = False, prefix: str = ""):
        """
        保存这一帧画面及其 YOLO 结果，转化为训练集格式。
        目录按 日期/小时 分层: images/YYYY-MM-DD/HH/ 与 labels/YYYY-MM-DD/HH/，
        每个小时一个子文件夹，避免截图堆在同一个目录。
        """
        self._ensure_dirs()

        # 按当地时间的 日期/小时 建子目录
        now = time.localtime()
        date_str = time.strftime("%Y-%m-%d", now)
        hour_str = time.strftime("%H", now)
        img_dir = os.path.join(self.img_dir, date_str, hour_str)
        lbl_dir = os.path.join(self.lbl_dir, date_str, hour_str)
        os.makedirs(img_dir, exist_ok=True)
        os.makedirs(lbl_dir, exist_ok=True)

        timestamp = int(time.time() * 1000)
        base_prefix = prefix if prefix else ("hard_" if is_hard_example else "auto_")
        filename = f"{base_prefix}{timestamp}"

        # 1. 保存图片
        img_path = os.path.join(img_dir, f"{filename}.jpg")
        try:
            cv2.imwrite(img_path, frame)
        except Exception as e:
            log.error(f"保存截图失败: {e}")
            return
            
        # 2. 保存 YOLO 归一化 txt 标签
        # 格式: class_id x_center y_center width height (所有值必须在 0.0 ~ 1.0 之间)
        lbl_path = os.path.join(lbl_dir, f"{filename}.txt")
        h, w = frame.shape[:2]
        
        try:
            with open(lbl_path, "w", encoding="utf-8") as f:
                # 遍历这一帧里的所有检测框
                for box in yolo_results.boxes:
                    cls_id = int(box.cls[0])
                    # 取出 xywh (中心点 x, 中心点 y, 宽, 高)
                    bx, by, bw, bh = box.xywh[0].cpu().numpy()
                    
                    # 归一化处理
                    norm_x = min(1.0, max(0.0, float(bx / w)))
                    norm_y = min(1.0, max(0.0, float(by / h)))
                    norm_w = min(1.0, max(0.0, float(bw / w)))
                    norm_h = min(1.0, max(0.0, float(bh / h)))
                    
                    # 类 id 原样保存 (010001010 为 0-9 多类; 4=Stump, 5=Slime 是合法怪类, 不能重映射)
                    f.write(f"{cls_id} {norm_x:.6f} {norm_y:.6f} {norm_w:.6f} {norm_h:.6f}\n")
                    
            if is_hard_example or prefix:
                log.info(f"[自动收集] 捕获特殊样本: {filename}.jpg")

        except Exception as e:
            log.error(f"保存标签失败: {e}")
            return None

        return img_path
            
    def maybe_save_heartbeat(self, frame, yolo_results):
        """定时心跳收集 (仅策略 C)：每隔 save_interval_seconds 保存一帧常规样本。
        不含测谎仪 / 困难样本逻辑，用于持续积累训练数据。"""
        now = time.time()
        if now - self.last_save_time > self.save_interval_seconds:
            saved = self.save_snapshot(frame, yolo_results, is_hard_example=False)
            self.last_save_time = now
            if saved:
                log.info(f"[心跳截图] 已保存 -> {saved}")

    def check_and_save(self, frame, yolo_results, has_captcha: bool = False):
        """
        主循环中调用的检查函数：决定当前这帧是否值得保存。
        依据：
        1. 是不是经过了设定的时间间隔？(常规心跳收集)
        2. 是不是出现了测谎仪等极端情况？(困难样本收集)
        """
        now = time.time()
        
        # --- 策略 A: 测谎仪突发 ---
        if has_captcha:
            # 加入一个 5 秒的冷却，防止一瞬间狂存几百张一样的测谎图
            if now - getattr(self, '_last_captcha_save', 0) > 5.0:
                self.save_snapshot(frame, yolo_results, is_hard_example=True, prefix="captcha_")
                self._last_captcha_save = now
            return

        # --- 策略 B: 困难样本分析 (低置信度) ---
        has_hard_boxes = False
        for box in yolo_results.boxes:
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])
            name = yolo_results.names[cls_id]
            # 如果认出了怪物(010001010 类 1-7)、玩家，但是信心度在 0.20 到 0.40 之间（半懂不懂）
            is_monster = 1 <= cls_id <= 7
            if (is_monster or name == "Player") and 0.20 <= conf <= 0.40:
                has_hard_boxes = True
                break
                
        if has_hard_boxes:
            if now - getattr(self, '_last_hard_save', 0) > 10.0:  # 困难样本 10s 冷却
                self.save_snapshot(frame, yolo_results, is_hard_example=True, prefix="hard_conf_")
                self._last_hard_save = now
            return
            
        # --- 策略 C: 佛系定时常规收集 ---
        if now - self.last_save_time > self.save_interval_seconds:
            self.save_snapshot(frame, yolo_results, is_hard_example=False)
            self.last_save_time = now
