@echo off
setlocal EnableDelayedExpansion

:: Set PyTorch C++ logging level to suppress info messages
set TORCH_CPP_LOG_LEVEL=ERROR

echo ==========================================
echo              Media Sorter
echo ==========================================
echo.

:: Check Python installation
python --version >nul 2>&1
if errorlevel 1 (
    echo  Python not found! Please install Python 3.10+
    echo    Download from: https://python.org
    pause
    exit /b 1
)

:: Check if data folder exists
if not exist "data" (
    echo  'data' folder not found!
    echo    Please create a 'data' folder and add your media files
    echo    Example: data\image1.jpg, data\video1.mp4, etc.
    pause
    exit /b 1
)

:: Count files in data folder
set /a file_count=0
for /r "data" %%f in (*.jpg *.jpeg *.png *.mp4 *.mov *.gif *.webp *.bmp *.avi *.mkv) do (
    set /a file_count+=1
)

if !file_count! EQU 0 (
    echo  No media files found in 'data' folder!
    echo    Supported formats: JPG, PNG, MP4, GIF, MOV, AVI, MKV, WEBP, BMP
    pause
    exit /b 1
)

echo  Found !file_count! media files to process
echo.

:: Create virtual environment if it doesn't exist
if not exist ".venv\Scripts\activate.bat" (
    echo  Creating virtual environment...
    python -m venv .venv
    if errorlevel 1 (
        echo  Failed to create virtual environment
        pause
        exit /b 1
    )
)

:: Activate virtual environment
echo  Activating virtual environment...
call .venv\Scripts\activate.bat

:: Upgrade pip and install dependencies
@REM echo  Installing/updating dependencies...
@REM python -m pip install --upgrade pip wheel
@REM python -m pip install -r requirements.txt --use-pep517

:: Check GPU availability
echo.
echo  Checking GPU availability...
python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"

:: Run the media sorter
echo.
echo  Starting media processing...
echo    This may take a while depending on the number of files
echo    Press Ctrl+C to stop at any time
echo.

python main.py

:: Check if processing was successful
if errorlevel 1 (
    echo.
    echo  Processing failed! Check the error messages above.
    echo    Common issues:
    echo    - Not enough GPU memory ^(try closing other applications^)
    echo    - Corrupted media files ^(check the data folder^)
    echo    - Missing dependencies ^(rerun this script^)
) else (
    echo.
    echo  Processing completed successfully!
    echo  Check the 'output' folder for sorted media
    
    :: Show basic statistics
    if exist "output" (
        echo.
        echo  Quick Statistics:
        for /d %%d in (output\*) do (
            set /a folder_count=0
            for %%f in ("%%d\*.*") do set /a folder_count+=1
            echo    %%~nxd: !folder_count! files
        )
    )
)

echo.
echo ==========================================
echo Press any key to exit...
pause >nul