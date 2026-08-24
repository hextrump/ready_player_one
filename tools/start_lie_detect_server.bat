@echo off
rem ============================================================
rem  hhh 训练机上启动远程测谎检测服务 (全远程架构的服务端)
rem  - 独立 lie-detector 项目 (用户昨天已跑通 opencv+samurai)
rem  - base_plus @ 512 (spike 实测 step ~70ms, 跟踪 perfect)
rem  - 日志追加到 ready_player_one/lie_detect_server.log
rem  - 从 ssh 用 `start` 启动; ssh 断开后服务仍存活
rem ============================================================
cd /d C:\Users\heyas\Documents\code\ready_player_one
set PYTHONIOENCODING=utf-8
"C:\Users\heyas\.cache\rimagination-notes\qwen3-asr-venv\Scripts\python.exe" tools\lie_detect_server.py --repo C:/Users/heyas/Documents/code/lie-detector >> "C:\Users\heyas\Documents\code\ready_player_one\lie_detect_server.log" 2>&1
