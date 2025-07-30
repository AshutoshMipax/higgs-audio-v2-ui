#!/usr/bin/env python3
"""
Python version validation script for Higgs Audio TTS
Ensures Python 3.10+ is being used
"""
import sys

def validate_python_version():
    """Validate that Python 3.10+ is being used"""
    major, minor = sys.version_info.major, sys.version_info.minor
    
    print(f"Current Python version: {major}.{minor}.{sys.version_info.micro}")
    print(f"Python executable: {sys.executable}")
    
    if (major, minor) < (3, 10):
        print(f"❌ ERROR: Python 3.10+ is required, but you have Python {major}.{minor}")
        print("Please install Python 3.10 or higher and recreate the virtual environment.")
        sys.exit(1)
    
    print(f"✅ Python version {major}.{minor} is compatible (3.10+ required)")
    return True

if __name__ == "__main__":
    validate_python_version()