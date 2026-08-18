@echo off
rem Passes every argument straight through: ai scan, ai system, ai doctor...
setlocal
set "ROOT=%~dp0.."
"%ROOT%\.venv\Scripts\python.exe" "%ROOT%\assistant.py" %*
