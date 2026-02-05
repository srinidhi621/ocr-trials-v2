# Azure Deployment Debugging Guide

## Problem: "Unexpected token '<', '< html>'" Error

This error means the Flask backend is returning an HTML error page instead of JSON. This typically happens when:
1. The container isn't running properly
2. The Flask app fails to start (import errors)
3. Azure infrastructure is returning a default error page

## Step-by-Step Debugging

### Step 1: Check Container Status

```bash
# For Container Apps:
az containerapp show \
  --name <your-app-name> \
  --resource-group <your-rg> \
  --query "properties.runningStatus"

# For App Service:
az webapp show \
  --name <your-app-name> \
  --resource-group <your-rg> \
  --query "state"
```

**Expected:** `"Running"` or `"Healthy"`  
**If not running:** Check deployment logs

### Step 2: View Real-time Logs

```bash
# For Container Apps:
az containerapp logs show \
  --name <your-app-name> \
  --resource-group <your-rg> \
  --follow

# For App Service:
az webapp log tail \
  --name <your-app-name> \
  --resource-group <your-rg>
```

**Look for:**
- ✓ "Starting OCR Pipeline UI on http://0.0.0.0:8000" (Flask started successfully)
- ✓ "✓ Azure credentials found"
- ✗ Python tracebacks or import errors
- ✗ "ModuleNotFoundError"
- ✗ "WARNING: Missing environment variables"

### Step 3: Test Health Endpoint

```bash
# Test if Flask is responding:
curl https://your-domain.azurewebsites.net/health
```

**Expected response:**
```json
{
  "status": "healthy",
  "version": "1.0",
  "timestamp": "2026-02-05T...",
  "directories": {
    "output": "True",
    "uploads": "True"
  }
}
```

**If this works:** Flask is running! The issue is in the `/run` endpoint.  
**If this fails:** Flask isn't starting at all.

### Step 4: Verify Environment Variables

In Azure Portal → Your App → **Configuration** → **Application Settings**:

Required variables:
```
AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT = https://your-resource.cognitiveservices.azure.com/
AZURE_DOCUMENT_INTELLIGENCE_KEY = your-key-here
PORT = 8000
```

### Step 5: Check Build Logs

When you run `az acr build`, check if the build succeeds:

```bash
az acr build \
  --registry <your-acr-name> \
  --image ocr-pipeline:latest \
  .
```

**Look for:**
- ✓ "Testing imports..." (should pass all 5 tests)
- ✗ Any import failures during build

### Step 6: Test Container Locally

**On Windows (PowerShell):**
```powershell
# Set environment variables
$env:AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT="https://your-endpoint.cognitiveservices.azure.com/"
$env:AZURE_DOCUMENT_INTELLIGENCE_KEY="your-key-here"

# Build and run container
docker build -t ocr-pipeline:test .
docker run -p 8000:8000 `
  -e AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT=$env:AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT `
  -e AZURE_DOCUMENT_INTELLIGENCE_KEY=$env:AZURE_DOCUMENT_INTELLIGENCE_KEY `
  ocr-pipeline:test
```

**On Mac/Linux:**
```bash
# Set environment variables
export AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT="https://your-endpoint.cognitiveservices.azure.com/"
export AZURE_DOCUMENT_INTELLIGENCE_KEY="your-key-here"

# Build and run container
docker build -t ocr-pipeline:test .
docker run -p 8000:8000 \
  -e AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT \
  -e AZURE_DOCUMENT_INTELLIGENCE_KEY \
  ocr-pipeline:test
```

Then test locally:
```bash
# Test health endpoint
curl http://localhost:8000/health

# Test file upload
curl -X POST -F "pdf=@test.pdf" -F "provider=azure" http://localhost:8000/run
```

### Step 7: SSH into Azure Container (if available)

```bash
# For App Service with SSH enabled:
az webapp ssh --name <your-app-name> --resource-group <your-rg>

# Once inside, test manually:
cd /app
python test_import.py
python -c "import app; print('App imported successfully')"
```

## Common Issues and Solutions

### Issue 1: ModuleNotFoundError
**Symptom:** Logs show "ModuleNotFoundError: No module named 'X'"  
**Solution:** Missing dependency in `requirements.txt`. Add it and rebuild.

### Issue 2: Azure Credentials Error
**Symptom:** "DefaultAzureCredential failed to retrieve a token"  
**Solution:** Use Key-based authentication (set `AZURE_DOCUMENT_INTELLIGENCE_KEY`)

### Issue 3: Permission Denied
**Symptom:** "PermissionError: [Errno 13] Permission denied: '/app/output'"  
**Solution:** Check Dockerfile ensures directories are created with proper permissions

### Issue 4: Port Binding Error
**Symptom:** "Address already in use"  
**Solution:** Make sure `PORT` environment variable matches Azure configuration (usually 8000)

### Issue 5: Timeout on Startup
**Symptom:** Container keeps restarting  
**Solution:** Increase startup timeout in Azure configuration, or optimize imports

## After Making Changes

1. **Rebuild the image:**
   ```bash
   az acr build --registry <your-acr> --image ocr-pipeline:latest .
   ```

2. **Restart the app:**
   ```bash
   # Container App:
   az containerapp revision restart --name <app-name> --resource-group <rg>
   
   # App Service:
   az webapp restart --name <app-name> --resource-group <rg>
   ```

3. **Clear browser cache** and test again

## Getting Help

If still stuck, collect these logs:
1. Container startup logs (from Step 2)
2. Build logs (from `az acr build`)
3. Health endpoint response (from Step 3)
4. Environment variables (without sensitive values)

Then share them for further debugging.
