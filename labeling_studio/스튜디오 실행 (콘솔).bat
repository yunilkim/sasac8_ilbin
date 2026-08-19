@echo off
rem  Thin shell only. Real logic lives in launch.ps1 (UTF-8, Korean OK).
rem  Keep this file ASCII-only: cmd.exe misparses UTF-8 Korean in .bat files.
chcp 65001 >nul
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0launch.ps1" -Console %*
