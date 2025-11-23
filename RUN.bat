@echo off
cd /d "%~dp0"
call venv\Scripts\activate
start "AI Meeting Lab" /min python meeting_app.py
exit
