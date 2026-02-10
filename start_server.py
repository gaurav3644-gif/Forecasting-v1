#!/usr/bin/env python3
"""
ForecastAI Server Startup Script
"""

from fastapi import FastAPI
import uvicorn
import sys

app = FastAPI()

@app.get("/")
def home():
    return "ForecastAI is running! Visit /forecast to start forecasting."

if __name__ == "__main__":
    print("🚀 Starting ForecastAI FastAPI Server...")
    print("📊 Visit http://localhost:8001 to access the application")
    print("Press Ctrl+C to stop the server")

    try:
        uvicorn.run(
            app,
            host="0.0.0.0",
            port=8002,
            log_level="info",
            reload=False
        )
    except KeyboardInterrupt:
        print("\n👋 Server stopped by user")
    except Exception as e:
        print(f"❌ Error starting server: {e}")
        sys.exit(1)