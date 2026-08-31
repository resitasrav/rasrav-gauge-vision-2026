@echo off
rem  Birlesik demoyu dogru yorumlayiciyla calistirir.
rem
rem  PowerShell'de "python demo\kos.py" yazinca PATH'teki SISTEM python'u
rem  (3.14) calisiyor ve orada ultralytics/torch/cv2 yok. Bu dosya yorumlayiciyi
rem  ACIKCA depo venv'inden seciyor. (kos.py kendini de duzeltiyor ama bu yol
rem  tek tikla calisiyor ve daha hizli.)
setlocal
set "KOK=%~dp0.."
set "PY=%KOK%\rasrav-gauge-vision-2026\.venv\Scripts\python.exe"

if not exist "%PY%" (
  echo HATA: sanal ortam bulunamadi: %PY%
  pause
  exit /b 1
)

"%PY%" "%~dp0kos.py" %*
echo.
pause
