# 🚀 Startup Modes - Cartoon Animation Studio

## 🎯 **Available Startup Modes**

The Cartoon Animation Studio supports multiple startup modes to prevent unwanted RunPod API connections and errors.

### 1. **🌐 Standalone Web Interface (Recommended)**

**Use this for local development and testing without RunPod connections.**

```bash
# Simple startup (no RunPod connections)
python start_standalone.py

# Or using the launcher
python launch.py web
```

**Features:**
- ✅ No RunPod API connections
- ✅ No 403 Forbidden errors  
- ✅ Web interface at http://localhost:7860
- ✅ Full animation and TTS generation
- ✅ Multi-character support

### 2. **🔌 API Server Mode**

**Use this for programmatic access without RunPod connections.**

```bash
# Start API server
python launch.py api

# Access at http://localhost:8000
# Documentation at http://localhost:8000/docs
```

**Features:**
- ✅ No RunPod API connections
- ✅ REST API endpoints
- ✅ JSON request/response format
- ✅ Base64 encoded outputs

### 3. **🔧 Standalone Worker Mode**

**Use this for testing the worker function without RunPod.**

```bash
# Start worker in standalone mode
python launch.py standalone
```

**Features:**
- ✅ No RunPod API connections
- ✅ Models loaded and ready
- ✅ Worker function available for testing
- ✅ No web interface (just the worker)

### 4. **🎬 RunPod Production Mode**

**Use this only when deploying to RunPod.**

```bash
# Only run this on RunPod infrastructure
python handler.py
```

**Features:**
- 🔄 RunPod serverless connections
- 🔄 RunPod API integration
- ⚠️ Will cause errors if run locally

## 🚫 **Preventing RunPod Connection Errors**

### The Problem
When running locally, you might see these errors:
```
2025-07-15 11:56:21 [info] {"env":"local","hostname":"8c5c0f533a87","level":"error","msg":"metrics.Workers err: http status code is 403","service":"aiapi","time":"2025-07-15T08:56:21Z","version":""}
```

### The Solution
**Always use standalone mode for local development:**

```bash
# ✅ CORRECT: Use standalone mode
python start_standalone.py

# ❌ WRONG: Direct handler execution (tries to connect to RunPod)
python handler.py
```

## 🔧 **Environment Variables**

The system uses these environment variables to control RunPod connections:

```bash
# Disable RunPod connections
export RUNPOD_STANDALONE_MODE=true
export STANDALONE_WORKER=true
export RUNPOD_DISABLE=true
```

These are automatically set by the standalone launchers.

## 🐳 **Docker Usage**

### Standalone Docker Container
```bash
# Build and run (uses standalone mode by default)
docker build -t cartoon-animation .
docker run -p 7860:7860 --gpus all cartoon-animation

# Container automatically uses start_standalone.py
# No RunPod connections will be made
```

### Force Different Modes in Docker
```bash
# Web interface mode
docker run -p 7860:7860 --gpus all cartoon-animation python launch.py web

# API server mode  
docker run -p 8000:8000 --gpus all cartoon-animation python launch.py api

# Standalone worker mode
docker run --gpus all cartoon-animation python launch.py standalone
```

## 🛠️ **Troubleshooting**

### Issue: RunPod 403 Errors
```
{"level":"error","msg":"metrics.Workers err: http status code is 403"}
```

**Solution:** Use standalone mode
```bash
python start_standalone.py  # ✅ Correct
```

### Issue: Missing Environment Variables
```
{"msg":"enve: missing optional envvar RUNPOD_GQL_API_URL"}
```

**Solution:** Environment variables are automatically set in standalone mode
```bash
# These are set automatically:
RUNPOD_STANDALONE_MODE=true
STANDALONE_WORKER=true  
RUNPOD_DISABLE=true
```

### Issue: Handler Tries to Connect to RunPod
**Solution:** Never run `handler.py` directly in local development
```bash
# ❌ DON'T DO THIS LOCALLY
python handler.py

# ✅ DO THIS INSTEAD
python start_standalone.py
```

## 📋 **Quick Reference**

| Mode | Command | Port | Use Case |
|------|---------|------|----------|
| **Standalone Web** | `python start_standalone.py` | 7860 | Local development |
| **Web Interface** | `python launch.py web` | 7860 | Interactive UI |
| **API Server** | `python launch.py api` | 8000 | Programmatic access |
| **Standalone Worker** | `python launch.py standalone` | - | Testing worker |
| **RunPod Production** | `python handler.py` | - | RunPod deployment only |

## 🎉 **Best Practices**

1. **Always use standalone mode for local development**
2. **Never run `handler.py` directly outside of RunPod**
3. **Use the launcher scripts instead of direct execution**
4. **Check environment variables if seeing connection errors**
5. **Use Docker with default CMD for hassle-free deployment**

## ✅ **Success Indicators**

When running correctly, you should see:
```bash
🎬 Cartoon Animation Studio - Standalone Mode
🔧 Environment configured for standalone operation
🚫 RunPod connections disabled
✅ Starting web interface...
🚀 Launching Cartoon Animation Web Interface...
📡 Interface will be available at: http://localhost:7860
💡 No RunPod connections will be made
```

**No 403 errors, no connection attempts, just pure animation generation!** 🎬✨ 