@echo off
chcp 65001 >nul
cls

echo ========================================================
echo    MEETING APP v0.6 - AUTOMATIC INSTALLER
echo ========================================================
echo.
echo This installer will download and install the latest
echo version of Meeting App from GitHub.
echo.
echo Repository: github.com/DarthSilent/NeuralSecretary
echo.
pause

setlocal enabledelayedexpansion

:: ==============================================
:: STEP 1: Check Python
:: ==============================================
echo.
echo [Step 1/8] Checking Python installation...
echo.

set PYTHON_CMD=
set PYTHON_OK=0

python --version >nul 2>&1
if !errorlevel! equ 0 (
    for /f "tokens=2" %%i in ('python --version 2^>^&1') do set PY_VER=%%i
    echo Found: Python !PY_VER!
    set PYTHON_CMD=python
    set PYTHON_OK=1
)

if "!PYTHON_OK!"=="0" (
    py --version >nul 2>&1
    if !errorlevel! equ 0 (
        for /f "tokens=2" %%i in ('py --version 2^>^&1') do set PY_VER=%%i
        echo Found: Python !PY_VER!
        set PYTHON_CMD=py
        set PYTHON_OK=1
    )
)

if "!PYTHON_OK!"=="0" (
    color 0C
    echo.
    echo [ERROR] Python 3.10+ not found!
    echo.
    echo Please install Python from: https://www.python.org/downloads/
    echo Make sure to check "Add Python to PATH" during installation.
    echo.
    pause
    exit /b 1
)

echo OK: Using !PYTHON_CMD!
echo.

:: ==============================================
:: STEP 2: Get Latest Release Info
:: ==============================================
echo [Step 2/8] Fetching latest release information...
echo.

set REPO=DarthSilent/NeuralSecretary
set API_URL=https://api.github.com/repos/%REPO%/releases/latest

:: Get latest release tag
powershell -Command "$response = Invoke-RestMethod -Uri '%API_URL%'; $response.tag_name" > latest_tag.tmp 2>nul

if not exist latest_tag.tmp (
    echo [ERROR] Failed to fetch release information from GitHub.
    echo Please check your internet connection.
    pause
    exit /b 1
)

set /p LATEST_TAG=<latest_tag.tmp
del latest_tag.tmp

echo Latest version: %LATEST_TAG%
echo.

:: Set download base URL
set DOWNLOAD_URL=https://github.com/%REPO%/releases/download/%LATEST_TAG%

:: ==============================================
:: STEP 3: Download Source Files
:: ==============================================
echo [Step 3/8] Downloading source files...
echo.

:: List of files to download
set FILES=meeting_app.py requirements.txt README.md USER_MANUAL.md LICENSE run_app.bat

for %%F in (%FILES%) do (
    echo Downloading %%F...
    powershell -Command "try { Invoke-WebRequest -Uri '%DOWNLOAD_URL%/%%F' -OutFile '%%F' } catch { Write-Host 'Warning: %%F not found in release' }"
)

:: Check critical files
if not exist meeting_app.py (
    echo.
    echo [ERROR] Failed to download meeting_app.py
    echo This file is required for the application to work.
    echo.
    echo Please download manually from:
    echo https://github.com/%REPO%/releases/latest
    echo.
    pause
    exit /b 1
)

if not exist requirements.txt (
    echo.
    echo [ERROR] Failed to download requirements.txt
    echo Creating default requirements.txt...
    (
        echo customtkinter^>=5.2.0
        echo sounddevice^>=0.4.6
        echo soundfile^>=0.12.1
        echo pydub^>=0.25.1
        echo torch^>=2.0.0
        echo torchaudio^>=2.0.0
        echo faster-whisper^>=0.10.0
        echo pyannote.audio^>=3.1.0
        echo scipy^>=1.11.0
        echo requests^>=2.31.0
        echo deepgram-sdk^>=3.0.0
        echo google-auth^>=2.23.0
        echo google-auth-oauthlib^>=1.1.0
        echo google-auth-httplib2^>=0.1.1
        echo google-api-python-client^>=2.100.0
        echo python-docx^>=1.0.0
        echo psutil^>=5.9.0
        echo GPUtil^>=1.4.0
        echo nvidia-cublas-cu12^>=12.1.0.26
        echo nvidia-cudnn-cu12^>=8.9.0.131
    ) > requirements.txt
)

echo.
echo Source files downloaded successfully.
echo.

:: ==============================================
:: STEP 4: Check Ollama (Optional)
:: ==============================================
echo [Step 4/8] Checking Ollama (optional)...
echo.

ollama --version >nul 2>&1
if !errorlevel! neq 0 (
    echo Ollama not found.
    echo.
    echo Ollama provides local LLM support (recommended but optional).
    echo Download from: https://ollama.com/download
    echo.
    choice /C YN /M "Continue without Ollama"
    if errorlevel 2 (
        echo.
        echo Please install Ollama and run this script again.
        pause
        exit /b 1
    )
) else (
    echo Ollama is installed.
    start /min ollama serve
)

echo.

:: ==============================================
:: STEP 5: Create Virtual Environment
:: ==============================================
echo [Step 5/8] Creating virtual environment...
echo.

if exist venv (
    echo Virtual environment already exists.
) else (
    !PYTHON_CMD! -m venv venv
    if !errorlevel! neq 0 (
        color 0C
        echo [ERROR] Failed to create virtual environment!
        pause
        exit /b 1
    )
    echo Virtual environment created.
)

call venv\Scripts\activate.bat

:: ==============================================
:: STEP 6: Upgrade pip
:: ==============================================
echo.
echo [Step 6/8] Upgrading pip...
python -m pip install --upgrade pip >nul 2>&1
echo Done.

:: ==============================================
:: STEP 7: Install Dependencies
:: ==============================================
echo.
echo [Step 7/8] Installing dependencies...
echo [WARNING] This may take 10-30 minutes (PyTorch, CUDA)
echo.

pip install -r requirements.txt

if !errorlevel! neq 0 (
    color 0C
    echo.
    echo [ERROR] Failed to install dependencies!
    pause
    exit /b 1
)

echo.
echo Dependencies installed.
echo.

:: ==============================================
:: STEP 8: Finalize Setup
:: ==============================================
echo [Step 8/8] Finalizing installation...
echo.

:: Create directories
if not exist Meeting_Records mkdir Meeting_Records
if not exist Voice_Samples mkdir Voice_Samples

:: Check FFmpeg
if not exist ffmpeg.exe (
    echo.
    echo [INFO] FFmpeg not found.
    echo FFmpeg is required for audio processing.
    echo.
    echo Download "ffmpeg-master-latest-win64-gpl.zip" from:
    echo https://github.com/BtbN/FFmpeg-Builds/releases
    echo.
    echo Extract ffmpeg.exe and ffprobe.exe to this directory.
    echo.
)

:: Create run script if not downloaded
if not exist run_app.bat (
    (
        echo @echo off
        echo cd /d "%%~dp0"
        echo call venv\Scripts\activate.bat
        echo python meeting_app.py
        echo pause
    ) > run_app.bat
)

:: Download Ollama model
ollama --version >nul 2>&1
if !errorlevel! equ 0 (
    echo.
    echo Ollama is available. Recommended model: qwen2.5:14b (7GB)
    echo.
    choice /C YN /M "Download qwen2.5:14b now"
    if errorlevel 1 (
        if not errorlevel 2 (
            echo.
            echo Downloading model...
            ollama pull qwen2.5:14b
        )
    )
)

:: ==============================================
:: COMPLETE
:: ==============================================
echo.
echo ========================================================
echo    INSTALLATION COMPLETE!
echo ========================================================
echo.
echo Downloaded version: %LATEST_TAG%
echo Installation directory: %CD%
echo.
echo IMPORTANT: Before first run
echo.
echo 1. Get HuggingFace token:
echo    https://huggingface.co/settings/tokens
echo.
echo 2. Accept model licenses:
echo    - https://huggingface.co/pyannote/speaker-diarization-3.1
echo    - https://huggingface.co/pyannote/wespeaker-voxceleb-resnet34-LM
echo.
echo 3. (Optional) Download FFmpeg if not already done
echo.
echo 4. Run: run_app.bat
echo.
echo 5. Enter your HF token in Settings
echo.
echo First run will download AI models (2-5 GB, one time only).
echo.
echo For detailed instructions, see USER_MANUAL.md
echo.
pause
