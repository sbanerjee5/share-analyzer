@echo off
echo ==========================================
echo Starting UK Share Analyzer - Backend
echo ==========================================
echo.

call venv\Scripts\activate
echo Virtual environment activated!
echo.

echo Starting FastAPI server...
python -m uvicorn main:app --reload

pause