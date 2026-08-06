# 🧬 Dataset Evolution Workflow

> **统一规范**: V13 标签 (0-3) + 1366×768 letterbox 画布.
> raw 标注、训练集、运行时抓帧三处完全一致, 无 remap, 无 resize 损耗.

## 📐 规范画布

| 项 | 值 |
|---|---|
| 分辨率 | **1366×768** (16:9) |
| letterbox 灰 | RGB(114,114,114) (YOLO 默认) |
| 标签格式 | YOLO normalized (cx cy w h), 0-1 |
| 类别 ID | 0=Player, 1=Monster, 2=Platform, 3=Rope |

**为什么 1366×768**: 匹配当前游戏窗口. 训练和推理看到同样的画布,
HP monitor / minimap / web 标注器都按 1366×768 像素坐标工作.

## 🏷️ V13 类别

| ID | Class | HTML 色 | BGR 色 |
|---|---|---|---|
| 0 | Player | `#0066cc` 蓝 | (255, 0, 0) |
| 1 | Monster | `#cc0000` 红 | (0, 0, 255) |
| 2 | Platform | `#00aa00` 绿 | (0, 255, 0) |
| 3 | Rope | `#00aaaa` 青 | (255, 255, 0) |

HP/MP 废弃 (改像素级检测).

## 🚀 4 阶段流程

### Stage 1: 数据采集 & 标注
```powershell
python tools/web_annotator.py    # http://localhost:8080
python tools/check_auto_dataset.py --limit 30   # 可视化检查
```
- 快捷键: `0/1/2/3` = Player/Monster/Platform/Rope, `D`=Delete, `S`=Save
- 位置: `data/auto_dataset/{images,labels}/`

### Stage 2: 构建训练集
```powershell
python scripts/build_dataset.py   [--ratio 0.8] [--seed 42] [--include-empty]
```
- 输入: `data/auto_dataset/` (1366×768 jpg + V13 txt)
- 输出: `data/super_brain_train/{images,labels}/{train,val}/` + `dataset.yaml`
- 校验: 类别 ID 越界直接报错退出

### Stage 3: 训练
```powershell
python train_super_brain.py
```
- yolov8n.pt, epochs=300, imgsz=960, batch=4, patience=50
- 输出: `runs/detect/super_brain/weights/best.pt`

### Stage 4: 部署
```powershell
cp runs/detect/super_brain/weights/best.pt models/super_brain.pt
```

## 🛠️ 常用操作

### 新截图进入数据集
1. 截图 → 放 `data/auto_dataset/images/`
2. `python tools/web_annotator.py` 标注
3. `python scripts/build_dataset.py && python train_super_brain.py`

### 想换游戏分辨率
- **改**: `src/utils/image_utils.py` 的 `CANONICAL_SIZE` + `src/capture/window_capture.py` 的 `CANONICAL_SIZE`
- **重做数据集**: `python scripts/resize_dataset.py --backup` (letterbox 全部原图到新尺寸)
- **labels 无需改** (0-1 归一化)

### 训练样本不均衡 (比如怪物多但 Player 少)
- `tools/check_auto_dataset.py --save --no-show` 抽样验证
- `scripts/build_dataset.py --include-empty` 加入负样本

## 📂 目录结构

```
data/
├── auto_dataset/              # 📌 唯一原始数据源 (1366×768 jpg + V13 labels)
│   ├── images/*.jpg
│   ├── images_orig/           # 备份 (resize 前原图, 可手动删除)
│   ├── labels/*.txt
│   └── map_ids.json           # 手工标注的 map_id (用于 meowdb 比对)
├── map_db/                    # meowdb 数据库 (gitignore)
│   ├── minimaps/{map_id}.png  # 406/417 张 minimap 缩略图 (138×116)
│   └── index.json             # 地图 ID 列表
└── super_brain_train/         # 自动生成, 可随时重建
    ├── images/{train,val}/
    ├── labels/{train,val}/
    └── dataset.yaml
```

## 📜 历史迁移

| 版本 | 变化 | 脚本 |
|---|---|---|
| V12 → V13 | 类别 ID 重映射 (2/3 HP/MP → 丢弃, 4→2, 5→3) | `scripts/migrate_labels.py` |
| 混合分辨率 → 1366×768 | letterbox 所有训练图到 1366×768 | `scripts/resize_dataset.py` |
| WSL 路径 → 相对路径 | 全部改相对项目根 | (已完成) |

两个迁移脚本都是**一次性的**, 当前 `auto_dataset/` 已经是 V13 + 1366×768 规范格式.

## 🗺️ meowdb 数据库 (B 方案)

**目标**: 利用 meowdb 的 minimap/怪物数据, 跟视觉模型比对, 评估 YOLO 准确度.

### 下载 + 构建数据库
```powershell
.venv/Scripts/python.exe scripts/build_map_db.py --all
# 输出: data/map_db/minimaps/*.png + index.json
```

### 手工标 map ID
**为什么手工**: 游戏 minimap 在 top-left (Artale 私服布局), 含动态内容
(player/NPC/怪物), 模板匹配/ORB 特征匹配分数都很低 (<0.5). 暂时靠人工标.

```powershell
.venv/Scripts/python.exe tools/web_map_id.py    # http://localhost:8081
```

## 🎯 单地图精细学习 (010001010)

**思路**: 不需要从游戏画面里抠, 直接用 meowdb 全图 + sprite 合成训练数据.

### 全图 + Sprite
```
data/map_db/
├── 010001010_full.png             # 全图 (2218x1870) - 整张地图背景
└── monsters/
    └── sprites/{id}_{Name}.png    # 7 种怪物的透明 sprite
```

### 合成数据生成
```powershell
.venv/Scripts/python.exe scripts/gen_yolo_010001010.py --count 300
.venv/Scripts/python.exe scripts/split_synthetic_010001010.py
```

**类别 (9 类)**:
| ID | Class | ID | Class |
|---|---|---|---|
| 0 | BlueSnail | 5 | OrangeMushroom |
| 1 | Shroom | 6 | GreenMushroom |
| 2 | RedSnail | 7 | Platform |
| 3 | Stump | 8 | Rope |
| 4 | Slime | | |

### 训练
```powershell
.venv/Scripts/python.exe train_010001010.py
# 输出: runs/detect/super_brain_010001010/weights/best.pt
```

**优势**:
- 不需要真实游戏帧 (你从另一台电脑拿的也能直接对比)
- 标签 100% 准确 (sprite 位置我们 100% 控制)
- 无限数据 (想生成多少张就多少)
- 怪物分类精细到具体种类 (不是 "Monster" 一类)

## 🛠️ 待办
- [x] 装 CUDA torch (rtx 2080 验证通过)
- [ ] 跑完 010001010 训练 (后台进行中)
- [ ] 在真实游戏帧上验证 mAP
- [ ] 标 map ID: 至少标 100 张覆盖主要地图
- [ ] 比对脚本: `scripts/compare_vision.py` (YOLO 输出 vs meowdb 真值)
