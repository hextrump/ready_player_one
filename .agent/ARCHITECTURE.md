# 🌌 Project Architecture: Ready Player One (v3.1)

This project is a game automation bot built on the **"Single Model Principle"**. Instead of multiple small models for different tasks, it uses a unified "Super Brain" for all entity detection.

## 🏗️ Core Philosophy
- **Single Model Principle (V12+ Unified)**: A single high-resolution YOLO model (1280x1280) detects **everything**: Player, Monsters, HP/MP bars, **Platforms**, and **Ropes**. This eliminates synchronization errors between "Brain" and "Terrain" models and drastically reduces GPU overhead.
- **Agentic-First Design**: The codebase is optimized for AI agents to comprehend, modify, and evolve.
- **Event-Driven**: Components communicate via the `GlobalBus` and `LocalBus` in `src/state/`.
- **Hybrid Data**: Combines real screenshots (Snapshots), Terrain labels, and synthetic data for robust training.

## 🧱 Component Breakdown

### 1. Vision & Perception (`src/perception/`)
- **`combat_brain.py`**: The central vision hub. It runs inference, filters detections, and maintains a "mental map" of targets.
- **`hp_monitor.py`**: Specifically tracks HP/MP segments for auto-healing.
- **`monster_tracker.py`**: Handles ID-to-Target mapping and persistence.

### 2. Decision & Brain (`src/brain/`)
- **`game_controller.py`**: State machine (FSM) that decides whether to Hunt, Patrol, Heal, or Rest.
- **`auto_healer.py`**: Logic for potion consumption based on vision data.

### 3. Navigation (`src/navigation/`)
- **`pathfinder.py`**: Simple A* or platform-based pathfinding.
- **`nav_mesh.py`**: Represents the platform structure of the current map.

### 4. Hardware Interaction (`src/capture/` & `src/navigation/action_translator.py`)
- **`window_capture.py`**: High-speed DXCAM or Window-handle based frame grabbing.
- **`action_translator.py`**: Converts logic "Move Left" into virtual key presses (`pydirectinput`).

## 📡 Communication Flow
1. `WindowCapture` -> Latest Frame.
2. `CombatBrain` -> inference (YOLO) -> `PerceptionData`.
3. `PerceptionData` -> `GlobalBus` (Broadcast).
4. `GameController` -> Consumes Events -> Decide Action.
5. `ActionTranslator` -> Key Press.
