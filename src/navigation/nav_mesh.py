"""
冒险岛 NavMesh (导航网格) 构建引擎
通过 YOLO 检测出的平台(Platform)和绳索(Rope)坐标，自动构建出可以用于 A* 寻路的有向图。

节点 (Node): 平台上的可用站立点，或绳索的两端
边 (Edge):
  - WALK: 同一平台上的相邻节点
  - CLIMB_UP / CLIMB_DOWN: 绳子连接的上下两端
  - JUMP: 跨平台的跳跃 (考虑跳跃高度和距离限制，且不能穿墙)
  - DOWN_JUMP: 下跳 (从一个平台穿透到正下方的平台)
"""

import math
import cv2
import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Optional

@dataclass
class Node:
    id: int
    x: int
    y: int
    platform_idx: int = -1

@dataclass
class Edge:
    n1: int
    n2: int
    cost: float
    action: str  # WALK, CLIMB_UP, CLIMB_DOWN, JUMP, DOWN_JUMP

class NavMeshBuilder:
    def __init__(
        self,
        node_spacing: int = 60,        # 节点间隔加大，减少路径碎片化
        max_jump_dx: int = 180,        # 最大横向跳跃距离 (对应冒险岛普通跳)
        max_jump_dy_up: int = -115,    # 最大向上跳跃高度 (约1.2个身位)
        max_jump_dy_down: int = 120,    # 最大向下跳跃落差
        max_down_jump_dy: int = 300,   # 下跳深度保留
        max_down_jump_dx: int = 50,
        rope_grab_dy: int = 125        # 跳起抓绳的最大高度 (脚底到绳底)
    ):
        self.node_spacing = node_spacing
        self.max_jump_dx = max_jump_dx
        self.max_jump_dy_up = max_jump_dy_up
        self.max_jump_dy_down = max_jump_dy_down
        self.max_down_jump_dy = max_down_jump_dy
        self.max_down_jump_dx = max_down_jump_dx
        self.rope_grab_dy = rope_grab_dy

        self.nodes: List[Node] = []
        self.edges: List[Edge] = []
        self.adjacency: Dict[int, List[Edge]] = {}

    def _intersects_platform(self, n1: Node, n2: Node, platforms: List) -> bool:
        """检查从 n1 到 n2 的连线是否穿透了其他不相关的平台实体 (防止穿墙跳跃)"""
        for p_idx, (px1, px2, py) in enumerate(platforms):
            if p_idx == n1.platform_idx or p_idx == n2.platform_idx:
                continue
            y_min, y_max = min(n1.y, n2.y), max(n1.y, n2.y)
            if y_min + 5 < py < y_max - 5:
                if n1.y != n2.y:
                    t = (py - n1.y) / (n2.y - n1.y)
                    ix = n1.x + t * (n2.x - n1.x)
                    if px1 <= ix <= px2:
                        return True
        return False

    def build_graph(self, platforms: List[tuple], ropes: List[tuple]):
        self.nodes.clear()
        self.edges.clear()
        self.adjacency.clear()

        p_nodes_map = {}

        # 1. 在平台上离散化生成 Nodes
        for p_idx, (x1, x2, y) in enumerate(platforms):
            xs = list(range(int(x1), int(x2), self.node_spacing))
            if int(x2) not in xs:
                xs.append(int(x2))
                
            # 把绳子经过的地方，或者绳子正下方允许跳跃抓取的地方打上 Node
            for (rx, ry1, ry2) in ropes:
                if x1 - 15 <= rx <= x2 + 15:
                    if y >= ry1 - 20 and y <= ry2 + self.rope_grab_dy:
                        if int(rx) not in xs:
                            xs.append(int(rx))
            
            xs = sorted(list(set(xs)))
            
            p_nodes_map[p_idx] = []
            for x in xs:
                cx = max(int(x1), min(int(x2), x))
                n = Node(len(self.nodes), cx, int(y), p_idx)
                self.nodes.append(n)
                p_nodes_map[p_idx].append(n)

        # 2. 同一平台的水平 WALK 连接
        for p_idx, p_nodes in p_nodes_map.items():
            for i in range(len(p_nodes) - 1):
                n1, n2 = p_nodes[i], p_nodes[i+1]
                dist = abs(n1.x - n2.x)
                self.add_edge(n1.id, n2.id, dist, "WALK")
                self.add_edge(n2.id, n1.id, dist, "WALK")

        # 3. 绳子 CLIMB 连接
        for (rx, ry1, ry2) in ropes:
            attached = []
            for n in self.nodes:
                # 寻找在这个绳索上方、交叉，或刚好在底部悬空跳抓范围内的平台点
                if abs(n.x - rx) < 15 and ry1 - 20 <= n.y <= ry2 + self.rope_grab_dy:
                    attached.append(n)
            
            attached.sort(key=lambda n: n.y) # 从上到下排序
            
            for i in range(len(attached) - 1):
                ntop = attached[i]
                nbot = attached[i+1]
                dist = abs(ntop.y - nbot.y)
                
                # 判断下面这个平台点是不是在悬挂绳子的空隙下方
                if nbot.y > ry2 + 15:
                    # 悬空绳索底部：得跳起来爬
                    self.add_edge(nbot.id, ntop.id, dist * 2.5, "JUMP_CLIMB")
                    # 爬到底部自然下落
                    self.add_edge(ntop.id, nbot.id, dist * 1.5, "DOWN_JUMP")
                else:
                    # 正常踩在绳底或绳索连贯经过
                    self.add_edge(nbot.id, ntop.id, dist * 2.0, "CLIMB_UP")
                    self.add_edge(ntop.id, nbot.id, dist * 2.0, "CLIMB_DOWN")

        # 4. 平台间 JUMP 和 DOWN_JUMP 连接
        for n1 in self.nodes:
            # 针对平台的端点额外考虑跨边缘跳跃
            p1_nodes = p_nodes_map.get(n1.platform_idx, [])
            is_edge_node = (n1 == p1_nodes[0] or n1 == p1_nodes[-1]) if p1_nodes else False

            for n2 in self.nodes:
                if n1.platform_idx == n2.platform_idx:
                    continue # 同一平台已建 WALK 边
                
                dx = n2.x - n1.x
                dy = n2.y - n1.y # 正数代表 n2 在下方
                adx = abs(dx)
                
                dist = math.hypot(dx, dy)
                
                # ==== A. 普通跳跃 (JUMP) ====
                if self.max_jump_dy_up <= dy <= self.max_jump_dy_down and adx <= self.max_jump_dx:
                    # 射线检测是否穿墙 (脑袋撞到顶/跳跃路径有阻挡)
                    if not self._intersects_platform(n1, n2, platforms):
                        # cost 惩罚: 尽量走平地少乱跳
                        self.add_edge(n1.id, n2.id, dist * 3.0, "JUMP")

                # ==== B. 下跳 (DOWN_JUMP) ====
                # 操作是 ↓ + Alt，只能垂直下落到下方的平台
                elif 0 < dy <= self.max_down_jump_dy and adx <= self.max_down_jump_dx:
                    # 不能穿透中途的平台（只允许落在正下方离得最近的第一块平台）
                    if not self._intersects_platform(n1, n2, platforms):
                        self.add_edge(n1.id, n2.id, dist * 1.5, "DOWN_JUMP")


    def add_edge(self, n1_id: int, n2_id: int, cost: float, action: str):
        e = Edge(n1_id, n2_id, cost, action)
        self.edges.append(e)
        if n1_id not in self.adjacency:
            self.adjacency[n1_id] = []
        self.adjacency[n1_id].append(e)

    def draw_debug_image(self, img_bgr: np.ndarray) -> np.ndarray:
        """把 NavMesh 完整渲染在原图上，不同颜色的边代表不同动作"""
        vis = img_bgr.copy()
        overlay = img_bgr.copy()

        colors = {
            "WALK": (255, 255, 255),       # 白: 走路
            "CLIMB_UP": (0, 255, 255),     # 黄: 爬上
            "CLIMB_DOWN": (0, 200, 200),   # 暗黄: 爬下
            "JUMP": (0, 165, 255),         # 橙: 跨越/上跳
            "DOWN_JUMP": (255, 0, 255)     # 紫: 穿透下跳
        }

        # 先画边 (先跳跃后行走，避免行走线被覆盖)
        for e in self.edges:
            if e.action == "WALK": continue
            n1, n2 = self.nodes[e.n1], self.nodes[e.n2]
            color = colors.get(e.action, (0, 255, 0))
            
            # 画稍微带弧度的箭头或者直线
            if e.action == "JUMP":
                # 计算抛物线控制点画曲线
                mx_c, my_c = (n1.x + n2.x)//2, min(n1.y, n2.y) - 30
                pts = np.array([
                    [n1.x, n1.y],
                    [mx_c, my_c],
                    [n2.x, n2.y]
                ], np.int32)
                cv2.polylines(overlay, [pts], False, color, 1)
                # 终点画个小圈
                cv2.circle(overlay, (n2.x, n2.y), 4, color, -1)
            else:
                cv2.arrowedLine(overlay, (n1.x, n1.y), (n2.x, n2.y), color, 1, cv2.LINE_AA, tipLength=0.04)

        for e in self.edges:
            if e.action != "WALK": continue
            n1, n2 = self.nodes[e.n1], self.nodes[e.n2]
            cv2.line(overlay, (n1.x, n1.y), (n2.x, n2.y), colors["WALK"], 2)

        # 叠加带有透明度的路线网格
        vis = cv2.addWeighted(overlay, 0.7, vis, 0.3, 0)

        # 最后画纯绿色的节点点位
        for n in self.nodes:
            cv2.circle(vis, (n.x, n.y), 3, (0, 255, 0), -1)

        # 统计图例 HUD
        stats = f"Nodes: {len(self.nodes)} | Edges: {len(self.edges)}"
        cv2.putText(vis, stats, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(vis, "White=Walk  Yellow=Climb  Orange=Jump  Purple=DownJump", (20, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        return vis


if __name__ == "__main__":
    from ultralytics import YOLO
    import sys

    # 这个脚本可以直接跑：把 YOLO 的结果喂给 NavMesh，然后保存渲染图
    img_path = sys.argv[1] if len(sys.argv) > 1 else "data/debug/yongshibuluo.png"
    model_path = r"runs\detect\runs\detect\terrain_v1\weights\best.pt"

    img = cv2.imread(img_path)
    if img is None:
        print(f"Error reading {img_path}")
        sys.exit(1)

    print(f"Running YOLO inference on {img_path}...")
    model = YOLO(model_path)
    results = model(img, conf=0.3)[0]

    raw_platforms = []
    raw_ropes = []

    for box in results.boxes:
        cls_id = int(box.cls[0])
        name = results.names[cls_id]
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        
        if name == 'Platform':
            # 存为 x_left, x_right, y_center
            raw_platforms.append((x1, x2, (y1 + y2) // 2))
        elif name == 'Rope':
            # 存为 x_center, y_top, y_bottom
            raw_ropes.append(((x1 + x2) // 2, y1, y2))

    print(f"YOLO Found: {len(raw_platforms)} Platforms, {len(raw_ropes)} Ropes.")

    # 构建 NavMesh
    builder = NavMeshBuilder()
    print("Building NavMesh Graph...")
    builder.build_graph(raw_platforms, raw_ropes)

    # 渲染可视化
    vis_img = builder.draw_debug_image(img)
    out_path = img_path.replace(".png", "_navmesh.png")
    cv2.imwrite(out_path, vis_img)
    print(f"NavMesh 渲染成功！图库节点数: {len(builder.nodes)}, 边数: {len(builder.edges)}")
    print(f"图片已保存至: {out_path}")
