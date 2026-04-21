@echo off
setlocal
set SCRIPT_DIR=%~dp0
cd /d "%SCRIPT_DIR%"
conda run -n bishe-oss python "%SCRIPT_DIR%web_demo.py"
endlocal
