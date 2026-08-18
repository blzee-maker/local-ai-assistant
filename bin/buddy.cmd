@echo off
rem Command Prompt entry point. cmd.exe has no idea the PowerShell profile
rem exists, so the assistant needs real files on PATH to be reachable here.
rem ROOT is derived from this script's own location, so moving the project
rem folder cannot silently break the shim.
setlocal
set "ROOT=%~dp0.."
"%ROOT%\.venv\Scripts\python.exe" "%ROOT%\assistant.py" wake %*
