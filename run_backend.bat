@echo off
cd /d d:\Dropbox\_Documents\_Vlance_2026\rtoy\tool_extract
set SERVER_DEBUG=0
set SERVER_RELOADER=0
if exist d:\Dropbox\_Documents\_Vlance_2026\rtoy\venv\Scripts\python.exe (
  d:\Dropbox\_Documents\_Vlance_2026\rtoy\venv\Scripts\python.exe server.py 1>server.out.log 2>server.err.log
) else if exist d:\Dropbox\_Documents\_Vlance_2026\rtoy\.venv\Scripts\python.exe (
  d:\Dropbox\_Documents\_Vlance_2026\rtoy\.venv\Scripts\python.exe server.py 1>server.out.log 2>server.err.log
) else (
  python server.py 1>server.out.log 2>server.err.log
)
