# 🌌 Project Architecture: Ready Player One (v7)

MapleStory auto-hunting bot. Detection = **YOLO V19 单类怪模型 + 名牌模板匹配定位玩家**;
决策 = **贪心战斗循环（无 NavMesh）**。决策结构参考 MapleStoryAutoLevelUp，
检测用更强的 YOLO。

## 🏗️ Core Philosophy
- **V19 Single-Class Model**: `models/monster_v19.pt` 是单类 YOLO（`{0: Monster}`）。
  专注怪检测，比旧的多类 "Super Brain" 更简单、更准，也省显存。
- **Player via Nametag**: V19 无 Player 类，玩家位置由 `src/perception/nametag_locator.py`
  模板匹配玩家静态名牌锚定，带合理性门控 + 漏检向画面中心衰减。
- **Greedy Decision (no NavMesh)**: 主循环"有怪在范围→打；否则朝最近怪贪心走；没怪→巡逻"。
  NavMesh / A* / 地形模型子系统已移除；多层靠跳发补刀 + 登台跳启发式 + 脱困跳兜底。
- **Eye-Hand Separation**: 后台视觉线程（~7fps）写共享感知缓存，主循环读缓存做动作，互不阻塞。

## 🧱 Component Breakdown

### 1. Capture (`src/capture/`)
- **`window_capture.py`**: 后台 BitBlt 抓帧、窗口查找与 resize。

### 2. Perception (`src/perception/`)
- **`hp_monitor.py`**: 追踪 HP/MP 条，供自动喝药。
- **`nametag_locator.py`**: 多尺度模板匹配玩家名牌 → 玩家坐标。

### 3. Brain (`src/brain/`)
- **`combat_brain.py`**: 中枢（感知 + 决策）。跑 V19 推理、锚定玩家位置、
  选目标（范围内优先，否则最近）、驱动 攻击/贪心靠近/巡逻，并启动后台视觉线程。
- **`game_controller.py`**: 底层后台键盘输入（AttachThreadInput + keybd_event）。
- **`auto_healer.py`**: 独立 HP/MP 药水线程，读 `hp_monitor` 血量。
- **`data_collector.py`**: 定期心跳截图保存，供模型再训练。

## 📡 Flow
1. `WindowCapture.grab()` → frame。
2. 感知线程: `CombatBrain.find_targets()` → V19 怪框 + 名牌玩家坐标 → 共享缓存。
3. 主循环: `select_target()` → 在攻击范围? `_attack()`（burst/跳发）:
   `_approach()`（贪心走 + 登台/下跳启发式 + 脱困跳）: 无怪 → `_patrol()`。
4. `GameController` 注入键盘扫描码。
5. `AutoHealer` 独立喝药。
