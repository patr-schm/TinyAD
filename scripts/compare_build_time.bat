@echo off
setlocal enabledelayedexpansion

:: Define variables
set REPO_DIR=%~dp0..
set BUILD_DIR=build-timing
set CSV_FILE=build_times.csv
set BUILD_TYPE=Release
set NUM_CORES=%NUMBER_OF_PROCESSORS%

:: Visual Studio environment setup
:: Detect Visual Studio installation (adjust paths as needed for your system)
if exist "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvarsall.bat" (
    set VS_ENV="C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvarsall.bat"
    set VS_VERSION=2022
) else if exist "C:\Program Files\Microsoft Visual Studio\2022\Professional\VC\Auxiliary\Build\vcvarsall.bat" (
    set VS_ENV="C:\Program Files\Microsoft Visual Studio\2022\Professional\VC\Auxiliary\Build\vcvarsall.bat"
    set VS_VERSION=2022
) else if exist "C:\Program Files\Microsoft Visual Studio\2022\Enterprise\VC\Auxiliary\Build\vcvarsall.bat" (
    set VS_ENV="C:\Program Files\Microsoft Visual Studio\2022\Enterprise\VC\Auxiliary\Build\vcvarsall.bat"
    set VS_VERSION=2022
) else if exist "C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\VC\Auxiliary\Build\vcvarsall.bat" (
    set VS_ENV="C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\VC\Auxiliary\Build\vcvarsall.bat"
    set VS_VERSION=2019
) else if exist "C:\Program Files (x86)\Microsoft Visual Studio\2019\Professional\VC\Auxiliary\Build\vcvarsall.bat" (
    set VS_ENV="C:\Program Files (x86)\Microsoft Visual Studio\2019\Professional\VC\Auxiliary\Build\vcvarsall.bat"
    set VS_VERSION=2019
) else if exist "C:\Program Files (x86)\Microsoft Visual Studio\2019\Enterprise\VC\Auxiliary\Build\vcvarsall.bat" (
    set VS_ENV="C:\Program Files (x86)\Microsoft Visual Studio\2019\Enterprise\VC\Auxiliary\Build\vcvarsall.bat"
    set VS_VERSION=2019
) else if exist "C:\Program Files (x86)\Microsoft Visual Studio\2017\Community\VC\Auxiliary\Build\vcvarsall.bat" (
    set VS_ENV="C:\Program Files (x86)\Microsoft Visual Studio\2017\Community\VC\Auxiliary\Build\vcvarsall.bat"
    set VS_VERSION=2017
) else (
    echo Error: Could not find Visual Studio vcvarsall.bat
    echo Please update the script with the correct path to your Visual Studio installation
    exit /b 1
)

:: Get branch names from command line or use defaults
if "%~1"=="" (
    set BRANCH1=main
) else (
    set BRANCH1=%~1
)

if "%~2"=="" (
    set BRANCH2=deferred-lambda-instantiation
) else (
    set BRANCH2=%~2
)

:: Special test mode
if /i "%BRANCH1%"=="test" (
    call :RunTestMode
    goto :eof
)

:: Initialize CSV file
echo Branch,Build Time (seconds) > %CSV_FILE%
echo Starting build time comparison between %BRANCH1% and %BRANCH2%...
echo Build configuration: %BUILD_TYPE% mode using Ninja with parallel builds (%NUM_CORES% cores)
echo Using Visual Studio %VS_VERSION% toolchain

:: Build first branch
call :BuildBranch %BRANCH1%
if errorlevel 1 goto :error

:: Build second branch
call :BuildBranch %BRANCH2%
if errorlevel 1 goto :error

:: Show results
echo.
echo ===================================
echo Results:
echo ===================================
echo.
type %CSV_FILE%
echo.
echo Build times written to %CSV_FILE%

goto :eof

:error
echo.
echo Error occurred during build process
exit /b 1

:RunTestMode
    echo Running in test mode - simulating builds
    echo Branch,Build Time (seconds) > %CSV_FILE%
    echo main,12.45 >> %CSV_FILE%
    echo feature,9.87 >> %CSV_FILE%
    echo.
    echo Test results:
    type %CSV_FILE%
    exit /b 0

:BuildBranch
    set BRANCH=%~1
    echo.
    echo ===================================
    echo Building branch: %BRANCH%
    echo ===================================
    echo.

    :: Checkout branch
    echo Checking out branch %BRANCH%...
    git -C "%REPO_DIR%" checkout %BRANCH%
    if errorlevel 1 (
        echo Error: Failed to checkout branch %BRANCH%
        exit /b 1
    )
    
    :: Clean and create build directory
    echo Cleaning previous build artifacts...
    if exist "%REPO_DIR%\%BUILD_DIR%" rmdir /s /q "%REPO_DIR%\%BUILD_DIR%"
    mkdir "%REPO_DIR%\%BUILD_DIR%"
    
    :: Set up Visual Studio environment (x64 architecture)
    echo Setting up Visual Studio %VS_VERSION% environment for x64 architecture...
    call %VS_ENV% x64
    if errorlevel 1 (
        echo Error: Failed to set up Visual Studio environment
        exit /b 1
    )
    
    :: Configure with CMake using Ninja and Release mode
    echo Running CMake with Ninja generator in %BUILD_TYPE% mode...
    cmake -S "%REPO_DIR%" -B "%REPO_DIR%\%BUILD_DIR%" -G "Ninja" -DCMAKE_BUILD_TYPE=%BUILD_TYPE% -DTINYAD_UNIT_TESTS=On
    if errorlevel 1 (
        echo Error: CMake configuration failed
        exit /b 1
    )
    
    :: Build and measure time
    echo Building with Ninja using %NUM_CORES% parallel jobs...
    set START_TIME=!time!
    
    cmake --build "%REPO_DIR%\%BUILD_DIR%" --config %BUILD_TYPE% -- -j %NUM_CORES%
    if errorlevel 1 (
        echo Error: Build failed
        exit /b 1
    )
    
    set END_TIME=!time!
    
    :: Calculate elapsed time
    call :CalculateElapsedTime "!START_TIME!" "!END_TIME!"
    echo %BRANCH%,!ELAPSED_TIME! >> %CSV_FILE%
    echo Build of %BRANCH% completed in !ELAPSED_TIME! seconds
    exit /b 0

:CalculateElapsedTime
    :: Get start and end times
    set START=%~1
    set END=%~2
    
    :: Parse hours, minutes, seconds, centiseconds
    for /f "tokens=1-4 delims=:,. " %%a in ("%START%") do (
        set /a START_H=%%a
        set /a START_M=%%b
        set /a START_S=%%c
        set /a START_CS=%%d
    )
    
    for /f "tokens=1-4 delims=:,. " %%a in ("%END%") do (
        set /a END_H=%%a
        set /a END_M=%%b
        set /a END_S=%%c
        set /a END_CS=%%d
    )
    
    :: Convert to centiseconds
    set /a START_TOTAL=(START_H*360000)+(START_M*6000)+(START_S*100)+START_CS
    set /a END_TOTAL=(END_H*360000)+(END_M*6000)+(END_S*100)+END_CS
    
    :: Handle midnight crossing
    if %END_TOTAL% LSS %START_TOTAL% set /a END_TOTAL+=8640000
    
    :: Calculate difference in seconds with 2 decimal places
    set /a DIFF=%END_TOTAL%-%START_TOTAL%
    set /a DIFF_SEC=%DIFF%/100
    set /a DIFF_CS=%DIFF%%%100
    
    if %DIFF_CS% LSS 10 (
        set ELAPSED_TIME=%DIFF_SEC%.0%DIFF_CS%
    ) else (
        set ELAPSED_TIME=%DIFF_SEC%.%DIFF_CS%
    )
    
    exit /b