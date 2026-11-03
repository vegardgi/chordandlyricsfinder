@echo off
rem Akkordverktøy – dobbeltklikk for å starte (Windows).
chcp 65001 >nul
setlocal
cd /d "%~dp0"
set "PATH=%USERPROFILE%\.local\bin;%PATH%"

set NEED=0
if not exist ".venv\Scripts\python.exe" set NEED=1
if not exist ".venv\.installert" set NEED=1
if "%NEED%"=="0" goto start

echo.
echo   ============================================================
echo     Første oppstart: installerer Akkordverktøy.
echo     Dette tar noen minutter (laster ned ca. 1-2 GB).
echo     Neste gang starter appen med en gang.
echo   ============================================================
echo.

where uv >nul 2>nul
if %errorlevel%==0 goto with_uv

set "PY="
py -3.12 -c "import sys" >nul 2>nul && set "PY=py -3.12"
if not defined PY py -3.11 -c "import sys" >nul 2>nul && set "PY=py -3.11"
if not defined PY python -c "import sys; sys.exit(not (3,10) <= sys.version_info[:2] <= (3,13))" >nul 2>nul && set "PY=python"
if defined PY goto with_python

echo * Fant ikke uv eller Python 3.10-3.13 - installerer uv ...
powershell -NoProfile -ExecutionPolicy Bypass -Command "irm https://astral.sh/uv/install.ps1 | iex"
set "PATH=%USERPROFILE%\.local\bin;%PATH%"
where uv >nul 2>nul || (echo ! Klarte ikke å installere uv. & goto fail)

:with_uv
echo * Lager Python-miljø med uv ...
uv venv --python 3.12 --allow-existing .venv || goto fail
echo * Installerer avhengigheter ...
uv pip install --python .venv\Scripts\python.exe -r requirements.txt || goto fail
goto installed

:with_python
echo * Lager Python-miljø med %PY% ...
%PY% -m venv .venv || goto fail
.venv\Scripts\python.exe -m pip install --upgrade pip
echo * Installerer avhengigheter ...
.venv\Scripts\python.exe -m pip install -r requirements.txt || goto fail

:installed
type nul > ".venv\.installert"
echo.
echo * Ferdig installert. Første låt laster i tillegg ned modellene (ca. 2 GB).

:start
.venv\Scripts\python.exe app.py
if errorlevel 1 goto fail
exit /b 0

:fail
echo.
echo ! Noe gikk galt. Se meldingene over.
pause
exit /b 1
