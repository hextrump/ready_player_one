"""
A* 寻路算法与动作翻译器
"""
import heapq
import cv2
import numpy as np
import time
from typing import List, Tuple, Dict
from src.navigation.nav_mesh import NavMeshBuilder, Node, Edge

class PathFinder:
    def __init__(self, nav_mesh: NavMeshBuilder):
        self.nav_mesh = nav_mesh

    def find_nearest_node(self, x: int, y: int) -> Node:
        """找到距离 (x,y) 最近的图节点"""
        best_node = None
        best_dist = float('inf')
        for n in self.nav_mesh.nodes:
            # 优先找同一水平线上的点，其次找垂直距离近的
            dx = n.x - x
            dy = n.y - y
            dist = (dx**2) + (dy**2 * 3.0) # Y轴距离惩罚加大
            if dist < best_dist:
                best_dist = dist
                best_node = n
        return best_node

    def get_path(self, start_x: int, start_y: int, target_x: int, target_y: int) -> List[Edge]:
        """A* 算法寻找最短路径"""
        start_node = self.find_nearest_node(start_x, start_y)
        target_node = self.find_nearest_node(target_x, target_y)
        
        if not start_node or not target_node:
            return []
            
        if start_node.id == target_node.id:
            return []

        # 优先队列存 (f_score, node_id)
        open_set = []
        heapq.heappush(open_set, (0, start_node.id))
        
        came_from: Dict[int, Edge] = {}
        
        g_score = {n.id: float('inf') for n in self.nav_mesh.nodes}
        g_score[start_node.id] = 0
        
        f_score = {n.id: float('inf') for n in self.nav_mesh.nodes}
        f_score[start_node.id] = self._heuristic(start_node, target_node)
        
        while open_set:
            _, current_id = heapq.heappop(open_set)
            
            if current_id == target_node.id:
                return self._reconstruct_path(came_from, current_id)
                
            for edge in self.nav_mesh.adjacency.get(current_id, []):
                neighbor_id = edge.n2
                tentative_g_score = g_score[current_id] + edge.cost
                
                if tentative_g_score < g_score[neighbor_id]:
                    came_from[neighbor_id] = edge
                    g_score[neighbor_id] = tentative_g_score
                    f_score[neighbor_id] = tentative_g_score + self._heuristic(self.nav_mesh.nodes[neighbor_id], target_node)
                    
                    # 取巧：如果它不在优先队列里，直接 push
                    # （标准做法应该更新优先级，但在 Python 里为了简单可以直接 push 新的）
                    heapq.heappush(open_set, (f_score[neighbor_id], neighbor_id))
                    
        return [] # no path found

    def _heuristic(self, n1: Node, n2: Node) -> float:
        """启发式函数：曼哈顿距离"""
        return abs(n1.x - n2.x) + abs(n1.y - n2.y) * 2.0

    def _reconstruct_path(self, came_from: Dict[int, Edge], current_id: int) -> List[Edge]:
        path = []
        while current_id in came_from:
            edge = came_from[current_id]
            path.append(edge)
            current_id = edge.n1
        path.reverse()
        return path

    def draw_path(self, frame: np.ndarray, path: List[Edge], start: tuple, end: tuple) -> np.ndarray:
        """把寻路结果画在图上"""
        vis = frame.copy()
        for i, edge in enumerate(path):
            n1 = self.nav_mesh.nodes[edge.n1]
            n2 = self.nav_mesh.nodes[edge.n2]
            
            # 画线
            color = (0, 0, 255) # 红色粗线代表选中路径
            if edge.action == "JUMP" or edge.action == "JUMP_CLIMB":
                cv2.line(vis, (n1.x, n1.y - 30), (n2.x, n2.y), color, 4)
            else:
                cv2.arrowedLine(vis, (n1.x, n1.y), (n2.x, n2.y), color, 4, tipLength=0.1)
                
            # 文字说明动作
            cv2.putText(vis, edge.action, ((n1.x+n2.x)//2, (n1.y+n2.y)//2 - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        cv2.circle(vis, start, 15, (255, 0, 0), -1) # 起点蓝色
        cv2.circle(vis, end, 15, (0, 255, 0), -1)   # 终点绿色
        cv2.putText(vis, "START", (start[0]-25, start[1]-20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,0), 2)
        cv2.putText(vis, "TARGET", (end[0]-30, end[1]-20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
        
        return vis


if __name__ == "__main__":
    from ultralytics import YOLO
    
    # ==== 测试寻路可视化 ====
    img_path = "data/debug/yongshibuluo.png"
    img = cv2.imread(img_path)
    model = YOLO(r"runs\detect\runs\detect\terrain_v1\weights\best.pt")
    res = model(img, conf=0.3)[0]
    
    plats, ropes = [], []
    for box in res.boxes:
        c = int(box.cls[0])
        n = res.names[c]
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        if n == 'Platform': plats.append((x1, x2, (y1+y2)//2))
        elif n == 'Rope': ropes.append(((x1+x2)//2, y1, y2))
        
    builder = NavMeshBuilder()
    builder.build_graph(plats, ropes)
    pf = PathFinder(builder)
    
    # 测试在勇士部落跑图: 从左下角跑到右上角的平台！
    start_pos = (500, 600)  # 左下角
    target_pos = (1200, 100) # 右上角
    
    st = time.time()
    path_edges = pf.get_path(*start_pos, *target_pos)
    print(f"A* 寻路耗时: {(time.time() - st)*1000:.2f}ms")
    
    print(" === 生成的动作序列 === ")
    for edge in path_edges:
        n1 = builder.nodes[edge.n1]
        n2 = builder.nodes[edge.n2]
        print(f" {edge.action:<10} => 从 ({n1.x},{n1.y}) 到 ({n2.x},{n2.y})")
        
    # 首先绘制底层的 NavMesh
    base_nav = builder.draw_debug_image(img)
    # 然后再画上红色的 A* 最短路径
    final_img = pf.draw_path(base_nav, path_edges, start_pos, target_pos)
    
    out = img_path.replace(".png", "_astar.png")
    cv2.imwrite(out, final_img)
    print(f"\n寻路结果图片保存在: {out}")
