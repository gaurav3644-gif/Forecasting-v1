# How to Package This Application

## Files/Folders to EXCLUDE when creating the ZIP:

**Do NOT include these:**
- `__pycache__/` (Python cache directories)
- `.env` (your personal environment variables with API keys!)
- `*.pyc` (compiled Python files)
- `venv/` or `env/` (virtual environment folders)
- `.git/` (git repository data)
- `.idea/` or `.vscode/` (IDE configuration)
- `*.log` (log files)
- `.DS_Store` (Mac system files)
- `sales_forecasting_data.csv` (any sample/test data files)
- `fix_retriever.py` (temporary fix script)
- Any uploaded user data files

## Files to INCLUDE:

**Essential Python files:**
- `app.py` - Main FastAPI application
- `features.py` - Feature engineering
- `forecasting.py` - Forecasting utilities
- `run_forecast2.py` - Forecast engine
- `retriever.py` - RAG system
- `assistant_openai.py` - AI assistant
- `assistant_context_packet.py` - Context building
- `on_demand_aggregator.py` - Data aggregation
- `supply_chain_planner.py` - Supply planning
- All other `.py` files

**Configuration files:**
- `requirements.txt` - Dependencies list
- `.env.example` - Environment variable template
- `README.md` - Setup and usage instructions
- `PACKAGING_INSTRUCTIONS.md` - This file

**Template files:**
- `templates/` folder with all `.html` files:
  - `base.html`
  - `upload.html`
  - `forecast.html`
  - `results_v2.html`
  - `supply_plan.html`
  - `error.html`

**Static assets (if any):**
- `static/` folder (CSS, JS, images if you have any)

## How to Create the ZIP File

### Option 1: Using File Explorer (Windows)

1. Open the application folder in File Explorer
2. Select all files EXCEPT the excluded ones listed above
3. Right-click → Send to → Compressed (zipped) folder
4. Name it: `forecasting_app.zip`

### Option 2: Using Command Line (Windows PowerShell)

```powershell
# Navigate to the parent folder of v3
cd "c:\Forecasting\v2\streamlit v2\fastapi_app"

# Create ZIP excluding unwanted files
Compress-Archive -Path "v3\*.py", "v3\templates", "v3\requirements.txt", "v3\README.md", "v3\.env.example" -DestinationPath "forecasting_app.zip" -Force
```

### Option 3: Using Command Line (Mac/Linux)

```bash
# Navigate to the parent folder
cd /path/to/forecasting/v2/streamlit v2/fastapi_app

# Create ZIP excluding unwanted files
zip -r forecasting_app.zip v3/ \
  -x "v3/__pycache__/*" \
  -x "v3/.env" \
  -x "v3/*.pyc" \
  -x "v3/venv/*" \
  -x "v3/.git/*" \
  -x "v3/*.log" \
  -x "v3/fix_retriever.py"
```

## What to Send Your Friend

1. **forecasting_app.zip** - The packaged application
2. **Installation instructions** - Point them to README.md inside the ZIP
3. **Optional**: If they don't have an OpenAI API key, let them know:
   - They can get one at https://platform.openai.com/api-keys
   - OR they can skip it and use all features except AI Assistant

## Security Reminder

**CRITICAL**:
- Never include your `.env` file in the ZIP (it contains your API key!)
- Never commit `.env` to git or share it
- Tell your friend to create their own `.env` file with their own API key

## After Your Friend Receives It

They should:
1. Extract the ZIP file
2. Open README.md and follow the "Installation & Setup" section
3. Install Python dependencies
4. (Optional) Create .env file with their OpenAI API key
5. Run the application

## Quick Test After Packaging

Before sending, test that it works:
1. Extract your ZIP to a temporary location
2. Create a new virtual environment
3. Follow the README.md instructions
4. Verify the app runs correctly
