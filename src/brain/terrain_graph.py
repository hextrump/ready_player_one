"""
TerrainGraph — 平台可达图 (世界树第一步: 图→树投影, 派生边算一次缓存)。

节点 = 平台 (y, x_left, x_right), 边 = 跳上/跳下 可跳关系。
构建后 support/_find_upper_platform 的 O(n) 扫描可变 O(1) 查表;
is_reachable/approach 用平台高度差判定 (不是怪中心), 是"隔层跳"规划的地基。

跳参数 (与 patrol_mover 对齐):
- 上跳: 高度差 <= EDGE_JUMP_MAX_DY, 目标平台 x 范围在源平台边缘可及
- 下跳: 目标更低 + x 范围重叠 (可直落, 无高度限制)
"""
from __future__ import annotations

from typing import Dict, List, Tuple

# 跳参数 (与 patrol_mover 一致)
EDGE_JUMP_MAX_DY = 140   # 单次上跳最大高度差 (px)
EDGE_JUMP_MAX_GAP = 60   # 上跳最大水平缝隙 (px)


class TerrainGraph:
    """平台可达图: 计算一次, 缓存跳上/跳下边。"""

    def __init__(self, platforms: List[tuple]):
        self.platforms = list(platforms)          # [(y, x_left, x_right)]
        self.index: Dict[tuple, int] = {}         # platform -> index
        self.adj: Dict[int, List[Tuple[int, str]]] = {}  # i -> [(j, 'up'|'down')]
        self._build()

    def _build(self):
        n = len(self.platforms)
        self.index = {p: i for i, p in enumerate(self.platforms)}
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                act = self._edge(self.platforms[i], self.platforms[j])
                if act:
                    self.adj.setdefault(i, []).append((j, act))

    @staticmethod
    def _edge(a, b):
        """a→b 是否可跳 (返回 'up'/'down'/None)。"""
        ya, xla, xra = a
        yb, xlb, xrb = b
        # 跳上: b 更高 (yb<ya), 高度差可跳, b 的 x 范围在 a 边缘可及
        if yb < ya and ya - yb <= EDGE_JUMP_MAX_DY:
            if (xlb <= xra + EDGE_JUMP_MAX_GAP and xrb >= xra) or \
               (xrb >= xla - EDGE_JUMP_MAX_GAP and xlb <= xla):
                return 'up'
        # 跳下: b 更低 + x 范围重叠 (可直落)
        if yb > ya and xlb < xra and xrb > xla:
            return 'down'
        return None

    def index_of(self, platform: tuple) -> int | None:
        """平台 → 节点 index (平台不在图中返回 None)。"""
        return self.index.get(platform)

    def reachable_from(self, i: int) -> List[Tuple[int, str]]:
        """从节点 i 可直接跳到的 [(j, action)]。"""
        return self.adj.get(i, [])

    def can_jump_to(self, i: int, j: int) -> bool:
        """i 能否一跳 (上或下) 到 j。"""
        return any(t == j for t, _ in self.adj.get(i, []))
