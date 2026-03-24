@echo off
setlocal
echo === RaveenCut ^| Build standalone executable ===
echo.

:: -- Check vendor binaries --------------------------------------------------
if not exist vendor\ffmpeg.exe (
    echo ERROR: vendor\ffmpeg.exe not found.
    echo.
    echo Download a Windows FFmpeg build ^(ffmpeg-master-latest-win64-gpl.zip^) from:
    echo   https://github.com/BtbN/FFmpeg-Builds/releases
    echo.
    echo Then place ffmpeg.exe and ffprobe.exe in the vendor\ folder and re-run.
    exit /b 1
)
if not exist vendor\ffprobe.exe (
    echo ERROR: vendor\ffprobe.exe not found.
    echo Place ffprobe.exe in the vendor\ folder and re-run.
    exit /b 1
)

:: -- Ensure PyInstaller is available ----------------------------------------
pip show pyinstaller >nul 2>&1
if errorlevel 1 (
    echo Installing PyInstaller...
    pip install pyinstaller
)

:: -- Build ------------------------------------------------------------------
echo Building...
pyinstaller RaveenCut.spec --clean --noconfirm
if errorlevel 1 (
    echo.
    echo Build FAILED.
    exit /b 1
)

echo.
echo Build complete.
echo Executable: dist\RaveenCut\RaveenCut.exe
endlocal
