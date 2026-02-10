@echo off
echo ========================================
echo  Git Verification and Commit Helper
echo ========================================
echo.

echo Step 1: Checking if templates folder exists...
if exist "templates\" (
    echo [OK] Templates folder found
    dir /b templates
) else (
    echo [ERROR] Templates folder not found!
    pause
    exit /b 1
)
echo.

echo Step 2: Checking Git status...
git status
echo.

echo Step 3: Would you like to add all files to Git? (Y/N)
set /p add_files="Add files to Git (Y/N)? "
if /i "%add_files%"=="Y" (
    echo Adding files to Git...
    git add *.py
    git add templates/
    git add requirements.txt
    git add .env.example
    git add README.md
    git add render.yaml
    git add Procfile
    git add runtime.txt
    git add .gitignore
    git add start_app.bat
    git add start_app.sh
    echo [OK] Files added
) else (
    echo Skipping file addition
)
echo.

echo Step 4: Checking what will be committed...
git status
echo.

echo Step 5: Would you like to commit these changes? (Y/N)
set /p commit_files="Commit changes (Y/N)? "
if /i "%commit_files%"=="Y" (
    git commit -m "Deploy-ready: Add all files including templates"
    echo [OK] Changes committed
) else (
    echo Skipping commit
)
echo.

echo Step 6: Verify templates are in Git...
echo.
git ls-files templates/
echo.

echo Step 7: Ready to push? (Y/N)
echo Note: Make sure you have set up your remote repository
set /p push_files="Push to remote (Y/N)? "
if /i "%push_files%"=="Y" (
    git push
    echo [OK] Pushed to remote
) else (
    echo Skipping push
    echo.
    echo To push manually later, run: git push
)
echo.

echo ========================================
echo Done! Check your GitHub/GitLab to verify
echo templates folder is there, then redeploy
echo on Render.
echo ========================================
pause
