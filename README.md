# Sales Forecasting Application

A comprehensive FastAPI-based forecasting application with AI-powered assistant, supply chain planning, and interactive visualizations.

## Features

- **Data Upload**: Upload CSV files with sales data
- **Forecasting**: Generate probabilistic forecasts using LightGBM quantile regression
- **Supply Chain Planning**: Inventory optimization with safety stock calculations
- **AI Assistant**: RAG-powered chat assistant for data insights (requires OpenAI API key)
- **Interactive Charts**: Plotly-based visualizations with AJAX updates

## Prerequisites

- Python 3.9 or higher
- OpenAI API key (optional, required only for AI Assistant feature)

## Installation & Setup

### 1. Extract the application

Extract the ZIP file to a folder of your choice.

### 2. Create a virtual environment (recommended)

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Mac/Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Set up environment variables (Optional - for AI Assistant)

Create a `.env` file in the application root with your OpenAI API key:

```
OPENAI_API_KEY=your-api-key-here
OPENAI_MODEL=gpt-4o-mini
OPENAI_MAX_TOKENS=500
OPENAI_TEMPERATURE=0.0
```

**Note**: The AI Assistant feature will not work without an OpenAI API key, but all other features (forecasting, supply planning, visualizations) work fine without it.

### 5. Run the application

```bash
uvicorn app:app --host 0.0.0.0 --port 8002 --reload
```

### 6. Access the application

Open your browser and navigate to:
- **Application**: http://localhost:8002
- **API Documentation**: http://localhost:8002/docs

## Usage Guide

### 1. Upload Data

- Go to the homepage (http://localhost:8002)
- Upload a CSV file with at least these columns:
  - `date` - Date column (YYYY-MM-DD format)
  - `sales` - Sales/demand values
  - Optional: `item`, `store`, and other feature columns

### 2. Generate Forecast

- After upload, click "Forecast" in the navigation
- Configure forecast parameters:
  - **Forecast Start Month**: Default is 3 months before last data month
  - **Number of Months**: Default is 7 months ahead
  - **Grain**: Select item/store for granular forecasts
  - **Additional Features**: Select columns to use as model features
  - **Seasonal Features**: Toggle seasonal patterns (Q4, pre-peak, etc.)
- Click "Generate Forecast"
- Wait for processing (progress bar will show status)

### 3. View Results

- After forecasting completes, click "Results"
- Use filters to view specific items/stores
- Toggle quantile visibility (P10, P30, P60, P90)
- Ask questions using the AI Assistant (if OpenAI key is configured)

### 4. Generate Supply Plan

- Click "Supply Plan" in navigation
- Configure inventory policies:
  - Lead time, MOQ, service level, etc.
- View optimized order quantities and inventory levels
- Identify stockout risks

## Sample Data Format

Your CSV should look like this:

```csv
date,sales,item,store,price,promo
2023-01-01,150,1,1,29.99,0
2023-01-02,165,1,1,29.99,1
2023-01-03,142,1,2,29.99,0
...
```

## Troubleshooting

### "Module not found" errors
- Make sure you activated the virtual environment
- Run `pip install -r requirements.txt` again

### "Port 8002 already in use"
- Change the port: `uvicorn app:app --port 8003`

### AI Assistant returns "Data not available"
- Make sure you set the OPENAI_API_KEY in .env file
- Restart the application after setting environment variables

### Forecast generation fails
- Check that your CSV has 'date' and 'sales' columns
- Ensure date column is in YYYY-MM-DD format
- Verify you have sufficient historical data (at least 30 days recommended)

## Application Architecture

- **Backend**: FastAPI (Python)
- **ML Model**: LightGBM quantile regression
- **AI Assistant**: OpenAI GPT with RAG (Retrieval-Augmented Generation)
- **Frontend**: Jinja2 templates with Bootstrap 5
- **Charts**: Plotly.js
- **Supply Planning**: Custom optimization algorithms

## Features Details

### Forecasting Engine
- Quantile regression for probabilistic forecasts (P10, P30, P60, P90)
- Automatic feature engineering (lags, rolling stats, seasonal features)
- Support for multi-item, multi-location forecasting
- Configurable seasonal patterns

### AI Assistant
- RAG-based system with semantic search
- Answers questions about your data, forecasts, and supply plans
- Prevents hallucination through strict validation
- Comprehensive knowledge of all items and stores

### Supply Chain Planning
- Inventory optimization with safety stock
- Lead time and MOQ constraints
- Service level targets
- Stockout risk identification

## Support

For issues or questions, check the logs in the terminal where the app is running.

## License

This application is provided as-is for educational and commercial use.
