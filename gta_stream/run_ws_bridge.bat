@echo off
cd /d "%~dp0"
py -3 gta_ws_bridge.py --host 127.0.0.1 --port 8765
