# Deploying to Render

This guide will help you deploy your Forecasting Application to Render.

## Prerequisites

1. A [Render account](https://render.com/) (free tier available)
2. Your code in a Git repository (GitHub, GitLab, or Bitbucket)

## Deployment Steps

### Option 1: Using render.yaml (Recommended)

1. **Push your code to GitHub**
   - Make sure `render.yaml` is in the root of your repository
   - Commit and push all files

2. **Create New Web Service on Render**
   - Go to [Render Dashboard](https://dashboard.render.com/)
   - Click "New +" → "Blueprint"
   - Connect your repository
   - Render will automatically detect `render.yaml`

3. **Set Environment Variables**
   - In Render dashboard, go to your service
   - Navigate to "Environment" tab
   - Add these variables:
     ```
     OPENAI_API_KEY=your_openai_api_key_here
     OPENAI_MODEL=gpt-4o-mini
     OPENAI_MAX_TOKENS=2000
     ```

4. **Deploy**
   - Render will automatically build and deploy your app
   - Your app will be available at: `https://your-app-name.onrender.com`

### Option 2: Manual Setup

1. **Create New Web Service**
   - Go to Render Dashboard
   - Click "New +" → "Web Service"
   - Connect your repository

2. **Configure Build Settings**
   - **Name**: `forecasting-app` (or your preferred name)
   - **Environment**: `Python 3`
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `uvicorn app:app --host 0.0.0.0 --port $PORT`

3. **Set Environment Variables** (same as Option 1)

4. **Deploy**
   - Click "Create Web Service"
   - Render will build and deploy

## Important Configuration Files

The following files are included for Render deployment:

- **`render.yaml`** - Render Blueprint configuration (Infrastructure as Code)
- **`Procfile`** - Process file for web server command
- **`runtime.txt`** - Specifies Python version
- **`requirements.txt`** - Python dependencies

## Environment Variables

Required:
- `OPENAI_API_KEY` - Your OpenAI API key (required for AI Assistant feature)

Optional:
- `OPENAI_MODEL` - Model to use (default: `gpt-4o-mini`)
- `OPENAI_MAX_TOKENS` - Max response tokens (default: `2000`)
- `OPENAI_TEMPERATURE` - Response temperature (default: `0.0`)

## Troubleshooting

### Error: "Could not import module 'main'"
- **Solution**: Make sure your start command is `uvicorn app:app --host 0.0.0.0 --port $PORT`
- The app is in `app.py`, not `main.py`

### Build Fails with Memory Error
- **Solution**: Some Python packages (like numpy, pandas) require more memory
- Upgrade to a paid Render plan with more memory
- Or optimize `requirements.txt` to use lighter versions

### App Times Out on Startup
- **Solution**: Increase the health check timeout in Render settings
- Large ML models (LightGBM) take time to load

### Static Files Not Loading
- **Solution**: Render free tier may have issues with static files
- Consider using a CDN or upgrading to a paid plan

## Free Tier Limitations

Render's free tier includes:
- 512 MB RAM
- Apps spin down after 15 minutes of inactivity
- 750 free hours per month per account

For production use, consider upgrading to a paid plan.

## Alternative Start Commands

If you need to customize the startup:

**With more workers:**
```bash
uvicorn app:app --host 0.0.0.0 --port $PORT --workers 2
```

**With SSL (if needed):**
```bash
uvicorn app:app --host 0.0.0.0 --port $PORT --ssl-keyfile /path/to/key.pem --ssl-certfile /path/to/cert.pem
```

## Post-Deployment

After successful deployment:
1. Test all features (upload, forecast, results, supply planning)
2. Monitor logs in Render dashboard
3. Set up alerts for failures
4. Consider adding custom domain

## Support

- [Render Documentation](https://render.com/docs)
- [Render Community](https://community.render.com/)
- [FastAPI Deployment Guide](https://fastapi.tiangolo.com/deployment/)
