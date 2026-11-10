@echo off
echo ==============================================
echo StreetCred AI - Startup Script
echo ==============================================

echo Starting ML Backend for Local App (Port 8000)...
start cmd /k "python -m uvicorn backend:app --host 127.0.0.1 --port 8000"

echo Starting ML Backend for Netlify Link (Port 5050)...
start cmd /k "python -m uvicorn backend:app --host 127.0.0.1 --port 5050"

echo Waiting for backends to initialize...
timeout /t 3 /nobreak

echo Opening the local web application...
start index.html

echo.
echo ==============================================
echo Backend is running! 
echo If you prefer the live web version, you can open:
echo https://streetcredx.netlify.app/
echo ==============================================
pause
