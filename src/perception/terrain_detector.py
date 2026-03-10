"""
TerrainDetector V1.0 — OpenCV 边缘检测提取冒险岛平台与绳索

算法流程:
1. 灰度 + 高斯模糊
2. Canny 边缘检测
3. HoughLinesP 霍夫直线检测
4. 过滤：水平线 → 平台候选, 垂直线 → 绳索候选
5. 聚类合并邻近线段
6. 输出可视化遮罩
"""

import cv2
import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple

@dataclass
class Platform:
    y: int          # 平台 Y 坐标（像素, 取平均）
    x_left: int     # 左边界
    x_right: int    # 右边界
    label: str = ""
    
    @property
    def width(self):
        return self.x_right - self.x_left
    
    @property
    def center_x(self):
        return (self.x_left + self.x_right) // 2

@dataclass
class Rope:
    x: int          # 绳子中心 X
    y_top: int      # 顶部
    y_bottom: int   # 底部
    label: str = ""
    
    @property
    def height(self):
        return self.y_bottom - self.y_top


class TerrainDetector:
    def __init__(
        self,
        # Canny 参数
        canny_low: int = 50,
        canny_high: int = 150,
        # Hough 参数
        hough_threshold: int = 40,
        min_line_length: int = 60,
        max_line_gap: int = 15,
        # 过滤参数
        horizontal_angle_thresh: float = 8.0,   # 水平线角度容差（度）
        vertical_angle_thresh: float = 15.0,     # 垂直线角度容差（度）
        min_platform_width: int = 50,            # 最小平台宽度
        min_rope_height: int = 40,               # 最小绳索高度
        # 聚类参数
        cluster_y_tolerance: int = 20,           # 同一平台的Y容差
        cluster_x_gap: int = 30,                 # 合并线段的X间距容差
        # ROI
        ui_top_cutoff: int = 60,                 # 屏幕顶部 UI 裁切
        ui_bottom_cutoff: int = 120,             # 屏幕底部 UI 裁切
    ):
        self.canny_low = canny_low
        self.canny_high = canny_high
        self.hough_threshold = hough_threshold
        self.min_line_length = min_line_length
        self.max_line_gap = max_line_gap
        self.horizontal_angle_thresh = horizontal_angle_thresh
        self.vertical_angle_thresh = vertical_angle_thresh
        self.min_platform_width = min_platform_width
        self.min_rope_height = min_rope_height
        self.cluster_y_tolerance = cluster_y_tolerance
        self.cluster_x_gap = cluster_x_gap
        self.ui_top_cutoff = ui_top_cutoff
        self.ui_bottom_cutoff = ui_bottom_cutoff

    def detect(self, frame: np.ndarray) -> Tuple[List[Platform], List[Rope]]:
        """
        从一张游戏截图中提取平台和绳子。
        """
        h, w = frame.shape[:2]
        
        # 1. 裁切 UI 区域（只分析游戏画面主体）
        roi = frame[self.ui_top_cutoff:h - self.ui_bottom_cutoff, :]
        roi_h, roi_w = roi.shape[:2]
        
        # 2. 灰度 + 模糊
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # 3. Canny 边缘检测
        edges = cv2.Canny(blurred, self.canny_low, self.canny_high)
        
        # 4. 霍夫直线检测
        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi / 180,
            threshold=self.hough_threshold,
            minLineLength=self.min_line_length,
            maxLineGap=self.max_line_gap,
        )
        
        if lines is None:
            return [], []
        
        horizontal_segments = []
        vertical_segments = []
        
        for line in lines:
            x1, y1, x2, y2 = line[0]
            # 还原到全图坐标
            y1 += self.ui_top_cutoff
            y2 += self.ui_top_cutoff
            
            dx = x2 - x1
            dy = y2 - y1
            length = np.sqrt(dx*dx + dy*dy)
            if length < 1:
                continue
                
            angle = abs(np.degrees(np.arctan2(dy, dx)))
            
            # 水平线检测
            if angle < self.horizontal_angle_thresh or angle > (180 - self.horizontal_angle_thresh):
                if abs(x2 - x1) >= self.min_platform_width:
                    xl, xr = min(x1, x2), max(x1, x2)
                    avg_y = (y1 + y2) // 2
                    horizontal_segments.append((xl, xr, avg_y))
            
            # 垂直线检测
            elif abs(angle - 90) < self.vertical_angle_thresh:
                if abs(y2 - y1) >= self.min_rope_height:
                    yt, yb = min(y1, y2), max(y1, y2)
                    avg_x = (x1 + x2) // 2
                    vertical_segments.append((avg_x, yt, yb))
        
        # 5. 聚类合并水平线段 → 平台
        platforms = self._cluster_horizontal(horizontal_segments)
        
        # 6. 聚类合并垂直线段 → 绳子
        ropes = self._cluster_vertical(vertical_segments)
        
        # 7. 按 Y 坐标排序（从上到下）
        platforms.sort(key=lambda p: p.y)
        ropes.sort(key=lambda r: r.y_top)
        
        # 8. 自动标记
        for i, p in enumerate(platforms):
            p.label = f"P{i}"
        for i, r in enumerate(ropes):
            r.label = f"R{i}"
        
        return platforms, ropes

    def _cluster_horizontal(self, segments) -> List[Platform]:
        """将 Y 坐标接近的水平线段合并成一个完整平台"""
        if not segments:
            return []
        
        # 按 Y 排序
        segments.sort(key=lambda s: s[2])
        
        clusters = []
        current_cluster = [segments[0]]
        
        for seg in segments[1:]:
            if abs(seg[2] - current_cluster[-1][2]) <= self.cluster_y_tolerance:
                current_cluster.append(seg)
            else:
                clusters.append(current_cluster)
                current_cluster = [seg]
        clusters.append(current_cluster)
        
        # 合并每个 cluster 为一个 Platform
        platforms = []
        for cluster in clusters:
            x_left = min(s[0] for s in cluster)
            x_right = max(s[1] for s in cluster)
            avg_y = int(np.mean([s[2] for s in cluster]))
            
            # 进一步合并非常近的子线段（中间有缺口但小于阈值）
            # 先排序再合并
            sorted_segs = sorted(cluster, key=lambda s: s[0])
            merged_ranges = []
            cur_left, cur_right = sorted_segs[0][0], sorted_segs[0][1]
            for seg in sorted_segs[1:]:
                if seg[0] <= cur_right + self.cluster_x_gap:
                    cur_right = max(cur_right, seg[1])
                else:
                    merged_ranges.append((cur_left, cur_right))
                    cur_left, cur_right = seg[0], seg[1]
            merged_ranges.append((cur_left, cur_right))
            
            for xl, xr in merged_ranges:
                if xr - xl >= self.min_platform_width:
                    platforms.append(Platform(y=avg_y, x_left=xl, x_right=xr))
        
        return platforms

    def _cluster_vertical(self, segments) -> List[Rope]:
        """将 X 坐标接近的垂直线段合并成一根绳子"""
        if not segments:
            return []
        
        # 按 X 排序
        segments.sort(key=lambda s: s[0])
        
        clusters = []
        current_cluster = [segments[0]]
        
        for seg in segments[1:]:
            if abs(seg[0] - current_cluster[-1][0]) <= 20:
                current_cluster.append(seg)
            else:
                clusters.append(current_cluster)
                current_cluster = [seg]
        clusters.append(current_cluster)
        
        ropes = []
        for cluster in clusters:
            avg_x = int(np.mean([s[0] for s in cluster]))
            y_top = min(s[1] for s in cluster)
            y_bottom = max(s[2] for s in cluster)
            
            if y_bottom - y_top >= self.min_rope_height:
                ropes.append(Rope(x=avg_x, y_top=y_top, y_bottom=y_bottom))
        
        return ropes

    def visualize(self, frame: np.ndarray, platforms: List[Platform], ropes: List[Rope]) -> np.ndarray:
        """
        在原图上绘制检测结果的半透明遮罩。
        - 红色半透明长条 = 平台
        - 黄色半透明竖条 = 绳子/梯子
        """
        vis = frame.copy()
        overlay = frame.copy()
        
        PLATFORM_COLOR = (0, 0, 255)      # 红色 (BGR)
        ROPE_COLOR = (0, 255, 255)         # 黄色
        PLATFORM_THICKNESS = 8
        
        # 画平台
        for p in platforms:
            # 半透明填充
            cv2.rectangle(overlay, 
                         (p.x_left, p.y - PLATFORM_THICKNESS//2), 
                         (p.x_right, p.y + PLATFORM_THICKNESS//2), 
                         PLATFORM_COLOR, -1)
            # 文字标签
            cv2.putText(vis, f"{p.label} y={p.y} w={p.width}", 
                       (p.x_left, p.y - 12),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, PLATFORM_COLOR, 1)
        
        # 画绳子
        for r in ropes:
            rope_w = 8
            cv2.rectangle(overlay,
                         (r.x - rope_w//2, r.y_top),
                         (r.x + rope_w//2, r.y_bottom),
                         ROPE_COLOR, -1)
            cv2.putText(vis, f"{r.label} h={r.height}",
                       (r.x - 20, r.y_top - 8),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, ROPE_COLOR, 1)
        
        # 半透明叠加
        result = cv2.addWeighted(overlay, 0.4, vis, 0.6, 0)
        
        return result


if __name__ == "__main__":
    import sys
    
    img_path = sys.argv[1] if len(sys.argv) > 1 else "data/debug/yongshibuluo.png"
    
    frame = cv2.imread(img_path)
    if frame is None:
        print(f"无法读取图片: {img_path}")
        sys.exit(1)
    
    print(f"图片尺寸: {frame.shape[1]}x{frame.shape[0]}")
    
    detector = TerrainDetector()
    platforms, ropes = detector.detect(frame)
    
    print(f"\n=== 检测到 {len(platforms)} 个平台 ===")
    for p in platforms:
        print(f"  {p.label}: y={p.y}, x=[{p.x_left} ~ {p.x_right}], width={p.width}px")
    
    print(f"\n=== 检测到 {len(ropes)} 根绳子/梯子 ===")
    for r in ropes:
        print(f"  {r.label}: x={r.x}, y=[{r.y_top} ~ {r.y_bottom}], height={r.height}px")
    
    # 保存可视化
    vis = detector.visualize(frame, platforms, ropes)
    out_path = img_path.replace('.png', '_terrain.png')
    cv2.imwrite(out_path, vis)
    print(f"\n可视化已保存到: {out_path}")
