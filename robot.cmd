@echo off
setlocal
set "APP_DIR=%~dp0"
pushd "%APP_DIR%"
if not exist ".venv\Scripts\python.exe" (
  echo Missing .venv\Scripts\python.exe
  exit /b 1
)
".venv\Scripts\python.exe" -m src.cli.entrypoint %*
set "CODE=%ERRORLEVEL%"
popd
exit /b %CODE%
