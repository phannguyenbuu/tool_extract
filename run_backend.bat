@echo off
set SERVER_DEBUG=1
set SERVER_RELOADER=1
set PYTHONUNBUFFERED=1
if exist venv\Scripts\python.exe (
  venv\Scripts\python.exe backend\server.py
) else if exist .venv\Scripts\python.exe (
  .venv\Scripts\python.exe backend\server.py
) else (
  python backend\server.py
)
