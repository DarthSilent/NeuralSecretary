@echo off
chcp 65001
echo ==========================================
echo      СБОРКА ИНСТАЛЛЯТОРА MEETING APP
echo ==========================================

echo.
echo [1/3] Очистка старых сборок...
if exist "build" rmdir /s /q "build"
if exist "dist" rmdir /s /q "dist"

echo.
echo [2/3] Запуск PyInstaller...
echo Это может занять несколько минут...
pyinstaller meeting_app.spec --noconfirm

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo ❌ Ошибка при сборке PyInstaller!
    pause
    exit /b %ERRORLEVEL%
)

echo.
echo [3/3] Сборка инсталлятора (Inno Setup)...
REM Пытаемся найти ISCC в стандартных путях
set "ISCC_PATH=C:\Program Files (x86)\Inno Setup 6\ISCC.exe"

if not exist "%ISCC_PATH%" (
    echo.
    echo ⚠️ Не найден компилятор Inno Setup по пути:
    echo "%ISCC_PATH%"
    echo.
    echo Если Inno Setup установлен в другом месте, отредактируйте этот батник.
    echo Или запустите компиляцию setup.iss вручную.
    pause
    exit /b 1
)

"%ISCC_PATH%" setup.iss

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo ❌ Ошибка при создании инсталлятора!
    pause
    exit /b %ERRORLEVEL%
)

echo.
echo ==========================================
echo ✅ ГОТОВО! Инсталлятор находится в папке Output (или рядом с скриптом).
echo ==========================================
pause
