@echo off
echo Setting up project with uv...

REM Install uv if not already installed (optional, remove if uv is pre-installed)
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
if %ERRORLEVEL% NEQ 0 (
    echo Error: Failed to install uv. Check your internet or PowerShell setup.
    exit /b %ERRORLEVEL%
)

REM Create virtual environment and install dependencies
uv venv
if %ERRORLEVEL% NEQ 0 (
    echo Error: Failed to create virtual environment.
    exit /b %ERRORLEVEL%
)
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