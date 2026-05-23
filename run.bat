@echo off
echo Setting up environment...
pip install -r requirements.txt
if %ERRORLEVEL% neq 0 (
    echo Error installing dependencies.
    pause
    exit /b
)

echo.
echo Checking for trained model...
if not exist "best.pt" (
    if not exist "runs\detect\train\weights\best.pt" (
        echo No custom model found. Would you like to train one now? [Y/N]
        set /p train_choice=
        if /i "%train_choice%"=="Y" (
            python train_model.py
        ) else (
            echo Using base YOLOv8n model...
        )
    )
)

echo Starting VisionCore Dashboard...
python app.py
pause
