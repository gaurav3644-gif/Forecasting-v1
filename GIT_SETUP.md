# Git Setup and Deployment Guide

This guide will help you set up Git properly and push your code to deploy on Render.

## Problem

Your templates folder isn't being deployed to Render because it's not committed to Git.

## Solution: Set Up Git Properly

### Step 1: Initialize Git Repository (if not already done)

```bash
cd "c:\Forecasting\v2\streamlit v2\fastapi_app\v3"
git init
```

### Step 2: Verify Templates Folder Exists

```bash
dir templates
```

You should see all your HTML files:
- base.html
- forecast.html
- index.html
- loading.html
- results.html
- results_v2.html
- signin.html
- signup.html
- supply_plan.html

### Step 3: Check Git Status

```bash
git status
```

If you see `templates/` in the untracked files or changes, that's good - we need to add it.

### Step 4: Add All Necessary Files

```bash
# Add all Python files
git add *.py

# Add templates folder (IMPORTANT!)
git add templates/

# Add configuration files
git add requirements.txt
git add .env.example
git add README.md
git add render.yaml
git add Procfile
git add runtime.txt
git add start_app.bat
git add start_app.sh
git add .gitignore
```

### Step 5: Verify Files Are Staged

```bash
git status
```

**IMPORTANT:** Make sure you see `templates/` listed in "Changes to be committed"

### Step 6: Commit Changes

```bash
git commit -m "Add all app files including templates for deployment"
```

### Step 7: Add Remote Repository (GitHub/GitLab)

If you haven't connected to GitHub yet:

```bash
# Replace with your repository URL
git remote add origin https://github.com/yourusername/your-repo.git
```

Or if already connected, update it:

```bash
git remote set-url origin https://github.com/yourusername/your-repo.git
```

### Step 8: Push to Remote

```bash
git branch -M main
git push -u origin main
```

## Verify on GitHub

1. Go to your GitHub repository
2. Navigate to the code
3. **Verify the `templates` folder is there with all HTML files**

## Deploy on Render

### Option 1: Auto-deploy (Recommended)

1. Go to [Render Dashboard](https://dashboard.render.com/)
2. If you already have a service, it should auto-deploy after the push
3. Watch the build logs to ensure it completes successfully

### Option 2: Manual Deploy

1. Go to your Render service
2. Click "Manual Deploy" → "Deploy latest commit"
3. Watch the build logs

## Troubleshooting

### If templates still not found:

**Check 1: Verify templates in Git repository**
```bash
git ls-files templates/
```
This should list all your template files. If empty, they weren't added.

**Check 2: Verify .gitignore isn't excluding templates**
```bash
type .gitignore
```
Make sure `templates/` is NOT in the .gitignore file.

**Check 3: Force add templates if needed**
```bash
git add -f templates/
git commit -m "Force add templates folder"
git push
```

### If Render build fails:

1. Check Render logs for specific errors
2. Verify Python version in `runtime.txt`
3. Ensure all dependencies in `requirements.txt` are correct

## Important Notes

✅ **DO commit:**
- All `.py` files
- `templates/` folder
- `requirements.txt`
- `render.yaml`, `Procfile`, `runtime.txt`
- `.env.example` (template only)
- `README.md`

❌ **DON'T commit:**
- `.env` (contains secrets!)
- `venv/` or `env/` folders
- `__pycache__/` folders
- `.pyc` files
- User data files (`.csv`, `.xlsx`)

## Quick Check Script

Run this to verify everything is ready:

```bash
# Check if templates exist
dir templates

# Check if they're in git
git ls-files templates/

# Check git status
git status
```

## After Successful Push

Your Render deployment should now work! Visit:
- Your app URL: `https://your-app-name.onrender.com`

The templates should load correctly.

## Still Having Issues?

If you still see the "TemplateNotFound" error:

1. Check Render build logs for the exact error
2. SSH into Render (if on paid plan) and verify:
   ```bash
   ls -la /opt/render/project/src/templates/
   ```
3. Ensure `app.py` has the correct templates path:
   ```python
   BASE_DIR = Path(__file__).resolve().parent
   templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))
   ```

## Need Help?

- Check GitHub repository to verify files are there
- Review Render build logs
- Ensure you pushed the latest changes
