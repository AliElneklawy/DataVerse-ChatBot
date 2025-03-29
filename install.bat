@echo off
echo Setting up project with uv...

REM Install uv if not already installed (optional, remove if uv is pre-installed)
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
if %ERRORLEVEL% NEQ 0 (
    echo Error: Failed to install uv. Check your internet or PowerShell setup.
    exit /b %ERRORLEVEL%
)

REM Create virtual environment
uv venv
if %ERRORLEVEL% NEQ 0 (
    echo Error: Failed to create virtual environment.
    exit /b %ERRORLEVEL%
)

REM Activate the virtual environment
call .venv\Scripts\activate.bat
if %ERRORLEVEL% NEQ 0 (
    echo Error: Failed to activate virtual environment.
    exit /b %ERRORLEVEL%
)
echo Virtual environment activated successfully.

REM Install audio dependencies for playsound in Windows
echo Installing audio dependencies for playsound...
powershell -ExecutionPolicy ByPass -Command "Add-Type -AssemblyName System.Speech" 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo Warning: Could not load System.Speech assembly. Some audio features may not work.
)

REM Install PyAudio which is often needed for audio-related packages
echo Installing PyAudio...
pip install PyAudio
if %ERRORLEVEL% NEQ 0 (
    echo Warning: Failed to install PyAudio directly. Trying alternative installation...
    
    REM Try with pipwin which can help with Windows binaries
    pip install pipwin
    if %ERRORLEVEL% EQU 0 (
        pip install pipwin
        pipwin install PyAudio
    ) else (
        echo Warning: Could not install PyAudio. You may need to install it manually.
    )
)

REM Now install main dependencies
uv sync
if %ERRORLEVEL% NEQ 0 (
    echo Error: Failed to install dependencies. Check pyproject.toml or uv setup.
    exit /b %ERRORLEVEL%
)

REM Run setup commands without needing manual venv activation
crawl4ai-setup
if %ERRORLEVEL% NEQ 0 (
    echo Error: Failed to run crawl4ai-setup.
    exit /b %ERRORLEVEL%
)
crawl4ai-doctor
if %ERRORLEVEL% NEQ 0 (
    echo Error: Failed to run crawl4ai-doctor.
    exit /b %ERRORLEVEL%
)

REM Handle Playwright browser installation if playwright is a dependency
playwright install
if %ERRORLEVEL% NEQ 0 (
    echo Error: Failed to install Playwright browsers.
    exit /b %ERRORLEVEL%
)

echo Installation and setup complete!
