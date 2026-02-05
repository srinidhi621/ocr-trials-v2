#!/usr/bin/env python3
"""
Test script to verify all imports work before starting Flask.
This helps diagnose startup issues.
"""
import sys
import traceback

def test_imports():
    """Test all imports that app.py depends on."""
    print("Testing imports...")
    
    try:
        print("1. Testing Flask...")
        from flask import Flask, request, jsonify
        print("   ✓ Flask OK")
    except Exception as e:
        print(f"   ✗ Flask FAILED: {e}")
        traceback.print_exc()
        return False
    
    try:
        print("2. Testing markdown...")
        import markdown
        print("   ✓ markdown OK")
    except Exception as e:
        print(f"   ✗ markdown FAILED: {e}")
        traceback.print_exc()
        return False
    
    try:
        print("3. Testing main.py imports...")
        from main import run_pipeline, generate_run_id
        print("   ✓ main.py OK")
    except Exception as e:
        print(f"   ✗ main.py FAILED: {e}")
        traceback.print_exc()
        return False
    
    try:
        print("4. Testing pipeline modules...")
        from pipeline.preprocessor import Preprocessor
        from pipeline.extractor import Extractor
        from pipeline.postprocessor import Postprocessor
        from pipeline.reviewer import Reviewer
        print("   ✓ pipeline modules OK")
    except Exception as e:
        print(f"   ✗ pipeline modules FAILED: {e}")
        traceback.print_exc()
        return False
    
    try:
        print("5. Testing providers...")
        from providers.azure_provider import AzureProvider
        print("   ✓ providers OK")
    except Exception as e:
        print(f"   ✗ providers FAILED: {e}")
        traceback.print_exc()
        return False
    
    print("\n✓ All imports successful!")
    return True

if __name__ == "__main__":
    success = test_imports()
    sys.exit(0 if success else 1)
