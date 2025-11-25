@echo off
chcp 65001 >nul
cls

echo ========================================================
echo    MEETING APP - COMPLETE INSTALLER
echo ========================================================
echo.

setlocal enabledelayedexpansion

:: ==============================================
:: STEP 1: Check Python
:: ==============================================
echo [Step 1/7] Checking Python installation...
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

echo Using: !PYTHON_CMD!
echo.

:: ==============================================
:: STEP 2: Check Ollama
:: ==============================================
echo [Step 2/7] Checking Ollama installation...
echo.

ollama --version >nul 2>&1
if !errorlevel! neq 0 (
    echo Ollama not found. Installing...
    echo.
    
    :: Download Ollama installer
    echo Downloading Ollama installer...
    powershell -Command "Invoke-WebRequest -Uri 'https://ollama.com/download/OllamaSetup.exe' -OutFile 'OllamaSetup.exe'"
    
    if exist OllamaSetup.exe (
        echo Installing Ollama...
        start /wait OllamaSetup.exe /S
        del OllamaSetup.exe
        
        :: Wait for Ollama service to start
        timeout /t 5 /nobreak >nul
        
        echo Ollama installed successfully.
    ) else (
        echo [WARNING] Could not download Ollama installer.
        echo Please install manually from: https://ollama.com/download
    )
) else (
    echo Ollama is already installed.
)

:: Start Ollama service
echo Starting Ollama service...
start /min ollama serve

echo.

:: ==============================================
:: STEP 3: Extract Program Files
:: ==============================================
echo [Step 3/7] Extracting program files...
echo.

:: Check if meeting_app.py exists
if not exist meeting_app.py (
    echo meeting_app.py not found. Downloading from Gist...
    powershell -Command "Invoke-WebRequest -Uri 'https://gist.githubusercontent.com/DarthSilent/d356769ed61e38501fd868c1d2745218/raw/meeting_app.py' -OutFile 'meeting_app.py'"
    
    if not exist meeting_app.py (
        echo [ERROR] Failed to download meeting_app.py!
        echo Please check your internet connection.
        pause
        exit /b 1
    ) else (
        echo Download successful.
    )
) else (
    echo meeting_app.py found.
)

:: Check if USER_MANUAL.md exists
if not exist USER_MANUAL.md (
    echo Downloading User Manual...
    powershell -Command "Invoke-WebRequest -Uri 'https://gist.githubusercontent.com/DarthSilent/d356769ed61e38501fd868c1d2745218/raw/USER_MANUAL.md' -OutFile 'USER_MANUAL.md'" >nul 2>&1
)

:: Create default settings.json
if not exist settings.json (
    echo Creating default settings.json...
    (
        echo {
        echo     "processing_mode": "local",
        echo     "hf_token": "",
        echo     "deepgram_key": "",
        echo     "cloud_use_mp3": true,
        echo     "keywords": "План, Сроки, Бюджет, API, Python",
        echo     "local_model_size": "large-v3",
        echo     "use_gdrive": false,
       echo     "keep_local": true,
        echo     "or_key": "",
        echo     "or_model": "deepseek/deepseek-r1-0528-qwen3-8b:free",
        echo     "system_prompt": "Твоя роль: профессиональный бизнес-ассистент и редактор.\\n1. ЗАДАЧА: Исправить ошибки транскрибации по контексту и оформить деловой протокол встречи.\\n2. ФОРМАТ (Markdown):\\n\\n# 📝 Протокол встречи от {date}\\n\\n## 🎯 Краткое содержание\\n(2–4 абзаца, суть обсуждения и итоги)\\n\\n## ✅ Принятые решения\\n* ...\\n\\n## 🛠 Задачи (Action Items)\\n| Задача | Ответственный | Срок |\\n|--------|---------------|------|\\n| ...    | ...           | ...  |\\n\\n## ❓ Открытые вопросы\\n* ...\\n\\nТон: деловой, спокойный, без воды.",
        echo     "input_device": "Default",
        echo     "rec_format": "mp3",
        echo     "local_compute": "int8",
        echo     "save_txt": true,
        echo     "save_docx": true,
        echo     "llm_provider": "local",
        echo     "local_model": "qwen2.5:14b",
        echo     "local_url": "http://localhost:11434/v1/chat/completions",
        echo     "current_prompt_name": "Стандартный (умный протокол^)",
        echo     "custom_prompts": {},
        echo     "batch_size": 8
        echo }
    ) > settings.json
)

echo.

:: ==============================================
:: STEP 4: Create Virtual Environment
:: ==============================================
echo [Step 4/7] Creating virtual environment...
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
)

call venv\Scripts\activate.bat

:: ==============================================
:: STEP 5: Upgrade pip
:: ==============================================
echo [Step 5/7] Upgrading pip...
echo.
python -m pip install --upgrade pip >nul 2>&1

:: ==============================================
:: STEP 6: Install Dependencies
:: ==============================================
echo [Step 6/7] Installing dependencies (this may take a while^)...
echo.

echo   - Core libraries...
pip install customtkinter sounddevice soundfile numpy scipy pillow pydub >nul 2>&1

echo   - Google Drive support...
pip install google-api-python-client google-auth-httplib2 google-auth-oauthlib >nul 2>&1

echo   - Document processing...
pip install python-docx lxml >nul 2>&1

echo   - PyTorch with CUDA support...
pip install torch==2.7.1+cu118 torchaudio==2.7.1+cu118 --index-url https://download.pytorch.org/whl/cu118

echo   - Torchcodec...
pip install torchcodec >nul 2>&1

echo   - Pyannote Audio...
pip install pyannote.audio==4.0.2 >nul 2>&1

echo   - Faster Whisper...
pip install faster-whisper >nul 2>&1

echo   - NVIDIA CUDA libraries...
pip install nvidia-cublas-cu12 nvidia-cudnn-cu12

echo   - HTTP client...
pip install requests >nul 2>&1

echo.
echo All dependencies installed successfully.
echo.

:: ==============================================
:: STEP 7: Setup and Finalize
:: ==============================================
echo [Step 7/7] Finalizing installation...
echo.

:: Check FFmpeg
if not exist ffmpeg.exe (
    echo [INFO] FFmpeg not found.
    echo FFmpeg is required for audio processing.
    echo Download from: https://github.com/BtbN/FFmpeg-Builds/releases
    echo Extract ffmpeg.exe and ffprobe.exe to this directory.
    echo.
)

:: Create directories
if not exist Meeting_Records mkdir Meeting_Records
if not exist Voice_Samples mkdir Voice_Samples

:: Create RUN.bat launcher
(
    echo @echo off
    echo chcp 65001 ^>nul
    echo cd /d "%%~dp0"
    echo if not exist venv (
    echo     echo [ERROR] Virtual environment not found!
    echo     echo Run install.bat first.
    echo     pause
    echo     exit /b 1
    echo ^)
    echo call venv\Scripts\activate.bat
    echo python meeting_app.py
    echo if errorlevel 1 pause
) > RUN.bat

:: Download Ollama models
echo.
echo Downloading AI models for Ollama (this will take time^)...
echo Recommended model: qwen2.5:14b
echo.
choice /C YN /M "Download qwen2.5:14b model now"
if errorlevel 2 goto skip_model
if errorlevel 1 (
    echo Downloading qwen2.5:14b...
    ollama pull qwen2.5:14b
)
:skip_model

:: ==============================================
:: COMPLETE
:: ==============================================
echo.
echo ========================================================
echo    INSTALLATION COMPLETE!
echo ========================================================
echo.
echo Next steps:
echo 1. If you haven't already, download FFmpeg (see above^)
echo 2. Get your Hugging Face token from: https://huggingface.co/settings/tokens
echo 3. Run RUN.bat to start the application
echo 4. Go to Settings and enter your HF token
echo.
echo The first run will download AI models (Whisper, Pyannote^).
echo This is normal and only happens once.
echo.
pause
