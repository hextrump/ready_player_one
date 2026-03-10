"""
V4.0 怪物追踪引擎 - 精灵图鉴模板匹配 (Sprite DB Template Matching)

彻底抛弃大模型坐标！直接从 maplestory.io 官方精灵图数据库加载完美带透明通道的怪物贴图，
用多尺度模板匹配在实机画面中定位怪物。

优势：
- 模板来自官方资源，零背景污染
- 支持多尺度匹配，适应游戏内不同大小
- 无需任何训练，无需 API 调用
"""

import cv2
import numpy as np
import os
import json
import time
from typing import List, Dict
from dataclasses import dataclass
from PIL import Image

from src.utils.logger import get_logger

log = get_logger("monster_tracker")

@dataclass
class TargetMonster:
    type_name: str
    x: int
    y: int
    w: int
    h: int
    confidence: float
    
    @property
    def center(self) -> tuple[int, int]:
        return (self.x + self.w // 2, self.y + self.h // 2)


class MonsterTracker:
    def __init__(self, match_threshold: float = 0.65):
        self.templates: Dict[str, List[np.ndarray]] = {}
        self.match_threshold = match_threshold
        self.db_dir = "data/monster_db"

    def load_from_db(self, monster_names: List[str]):
        """
        从精灵图鉴数据库加载指定怪物的模板。
        monster_names: 怪物英文名列表，如 ["Wild Boar", "Dark Axe Stump"]
        """
        index_path = os.path.join(self.db_dir, "monster_index.json")
        if not os.path.exists(index_path):
            log.error("monster_index.json not found!")
            return
            
        with open(index_path, "r", encoding="utf-8") as f:
            index = json.load(f)
        
        for target_name in monster_names:
            target_lower = target_name.lower()
            
            for mob_id, info in index.items():
                if info['name'].lower() == target_lower:
                    png_path = info.get('png')
                    if not png_path or not os.path.exists(png_path):
                        continue
                    
                    # 加载精灵图 (RGBA，带透明通道)
                    sprite_rgba = cv2.imread(png_path, cv2.IMREAD_UNCHANGED)
                    if sprite_rgba is None:
                        continue
                    
                    # 如果有 alpha 通道，提取出来作为 mask
                    if sprite_rgba.shape[2] == 4:
                        bgr = sprite_rgba[:, :, :3]
                        alpha = sprite_rgba[:, :, 3]
                        # alpha > 128 的像素是前景（怪物本体）
                        _, mask = cv2.threshold(alpha, 128, 255, cv2.THRESH_BINARY)
                    else:
                        bgr = sprite_rgba
                        mask = np.ones(bgr.shape[:2], dtype=np.uint8) * 255
                    
                    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
                    
                    if target_name not in self.templates:
                        self.templates[target_name] = []
                    
                    # 游戏里的怪物可能比精灵图大或小，所以我们生成多尺度模板
                    # 精灵图通常是 1x，游戏里大约是 1.0x ~ 2.0x
                    for scale in [0.8, 1.0, 1.2, 1.5, 1.8, 2.0]:
                        new_w = int(gray.shape[1] * scale)
                        new_h = int(gray.shape[0] * scale)
                        if new_w < 15 or new_h < 15:
                            continue
                        scaled_gray = cv2.resize(gray, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
                        scaled_mask = cv2.resize(mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
                        self.templates[target_name].append((scaled_gray, scaled_mask, scale))
                    
                    log.info(f"Loaded sprite for [{target_name}] (ID:{mob_id}), generated 6 scale variants.")
                    break
            else:
                log.warning(f"Monster [{target_name}] not found in database!")
        
        total = sum(len(v) for v in self.templates.values())
        log.info(f"Template library ready: {len(self.templates)} types, {total} total templates.")

    def scan(self, frame: np.ndarray) -> List[TargetMonster]:
        found_monsters = []
        if not self.templates:
            return found_monsters
            
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        for m_type, tpl_list in self.templates.items():
            best_for_type = []
            
            for tpl_gray, tpl_mask, scale in tpl_list:
                th, tw = tpl_gray.shape[:2]
                
                if th >= frame_gray.shape[0] or tw >= frame_gray.shape[1]:
                    continue
                
                # TM_SQDIFF_NORMED: 0 = perfect match, 1 = no match
                res = cv2.matchTemplate(frame_gray, tpl_gray, cv2.TM_SQDIFF_NORMED, mask=tpl_mask)
                
                diff_threshold = 1.0 - self.match_threshold
                loc = np.where(res <= diff_threshold)
                
                pts = list(zip(*loc[::-1]))
                pts_scored = [(pt, float(res[pt[1], pt[0]])) for pt in pts]
                pts_scored.sort(key=lambda x: x[1])
                
                for pt, score in pts_scored[:50]:
                    conf = 1.0 - score
                    best_for_type.append(
                        TargetMonster(type_name=m_type, x=pt[0], y=pt[1], w=tw, h=th, confidence=conf)
                    )
            
            # NMS per type
            best_for_type = self._nms(best_for_type)
            found_monsters.extend(best_for_type)
                    
        return found_monsters

    def _nms(self, items: List[TargetMonster], overlap_thresh=0.3) -> List[TargetMonster]:
        if not items:
            return []
            
        items = sorted(items, key=lambda i: i.confidence, reverse=True)
        keep = []
        
        for itm in items:
            overlap = False
            for k in keep:
                dx = abs(itm.center[0] - k.center[0])
                dy = abs(itm.center[1] - k.center[1])
                min_w = min(itm.w, k.w)
                min_h = min(itm.h, k.h)
                if dx < min_w * (1 - overlap_thresh) and dy < min_h * (1 - overlap_thresh):
                    overlap = True
                    break
            if not overlap:
                keep.append(itm)
                
        return keep
