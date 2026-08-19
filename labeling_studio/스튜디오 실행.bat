@echo off
rem  Thin shell only. Real logic lives in launch.ps1 (UTF-8, Korean OK).
rem  Keep this file ASCII-only: cmd.exe misparses UTF-8 Korean in .bat files.
rem  chcp is needed here too: first run prints setup progress in Korean.
chcp 65001 >nul
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0launch.ps1" %*
