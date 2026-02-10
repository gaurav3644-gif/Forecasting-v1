#!/usr/bin/env python3
"""
Debug script for ForecastAI app loading issue
Run this to test the complete flow and identify where it gets stuck
"""

import requests
import time
import json

BASE_URL = "http://localhost:8000"

def test_upload():
    """Test file upload"""
    print("1. Testing file upload...")
    try:
        with open("../store_sales.csv", "rb") as f:
            files = {"file": f}
            response = requests.post(f"{BASE_URL}/upload", files=files)
        print(f"   Status: {response.status_code}")
        if response.status_code == 303:
            print("   ✓ Upload successful, redirected to forecast page")
            return True
        else:
            print(f"   ✗ Upload failed: {response.text}")
            return False
    except Exception as e:
        print(f"   ✗ Upload error: {e}")
        return False

def test_status():
    """Check session status"""
    print("2. Checking session status...")
    try:
        response = requests.get(f"{BASE_URL}/status")
        if response.status_code == 200:
            status = response.json()
            print(f"   Session exists: {status['session_exists']}")
            print(f"   Data shape: {status['data_shape']}")
            print(f"   Horizon: {status['horizon']}")
            return status['session_exists']
        else:
            print(f"   ✗ Status check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"   ✗ Status check error: {e}")
        return False

def test_forecast_form():
    """Test forecast form submission"""
    print("3. Testing forecast form submission...")
    try:
        data = {"horizon": 30}
        response = requests.post(f"{BASE_URL}/forecast", data=data)
        print(f"   Status: {response.status_code}")
        if response.status_code == 303:
            print("   ✓ Form submission successful, redirected to loading")
            return True
        else:
            print(f"   ✗ Form submission failed: {response.text}")
            return False
    except Exception as e:
        print(f"   ✗ Form submission error: {e}")
        return False

def test_forecast_generation():
    """Test forecast generation"""
    print("4. Testing forecast generation...")
    try:
        start_time = time.time()
        response = requests.post(f"{BASE_URL}/generate_forecast",
                               headers={"Content-Type": "application/json"},
                               timeout=60)
        elapsed = time.time() - start_time
        print(".1f")
        print(f"   Status: {response.status_code}")

        if response.status_code == 200:
            result = response.json()
            print(f"   Response: {result}")
            if result.get("status") == "success":
                print("   ✓ Forecast generation successful")
                return True
            else:
                print(f"   ✗ Forecast generation failed: {result.get('message')}")
                return False
        else:
            print(f"   ✗ Request failed: {response.text}")
            return False
    except requests.exceptions.Timeout:
        print("   ✗ Request timed out after 60 seconds")
        return False
    except Exception as e:
        print(f"   ✗ Forecast generation error: {e}")
        return False

def main():
    print("🔍 ForecastAI Debug Script")
    print("=" * 50)

    # Test each step
    upload_ok = test_upload()
    if not upload_ok:
        print("\n❌ Issue: File upload failed")
        return

    time.sleep(1)  # Brief pause

    status_ok = test_status()
    if not status_ok:
        print("\n❌ Issue: Session data not found after upload")
        return

    time.sleep(1)  # Brief pause

    form_ok = test_forecast_form()
    if not form_ok:
        print("\n❌ Issue: Forecast form submission failed")
        return

    time.sleep(1)  # Brief pause

    forecast_ok = test_forecast_generation()
    if not forecast_ok:
        print("\n❌ Issue: Forecast generation failed or timed out")
        return

    print("\n✅ All tests passed! The app should work correctly.")
    print("If you're still seeing the loading page stuck, check:")
    print("1. Browser developer tools (F12) for JavaScript errors")
    print("2. Server terminal output for error messages")
    print("3. Try with a smaller CSV file (first 1000 rows)")

if __name__ == "__main__":
    main()