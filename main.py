"""
Main entry point for deployment platforms that expect main.py
This simply imports the FastAPI app from app.py
"""
from app import app

__all__ = ["app"]
