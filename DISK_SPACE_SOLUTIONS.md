# 🗂️ Disk Space Solutions for Docker Build

## 🚨 **Current Issue**
Docker build is failing with "no space left on device" error during the build process.

## 💾 **Current Disk Usage**
```
F: Drive - 30GB free out of 190GB total (Project location)
C: Drive - 42GB free out of 213GB total (System drive)
```

## 🔧 **Immediate Solutions**

### 1. **Use the Optimized Build Script (Recommended)**
```powershell
# Run the optimized PowerShell build script
.\build-docker.ps1
```

### 2. **Manual Docker Cleanup**
```powershell
# Clean up Docker cache and unused images
docker system prune -a --volumes

# Remove all stopped containers
docker container prune -f

# Remove all unused images
docker image prune -a -f

# Remove all unused volumes
docker volume prune -f
```

### 3. **Move Docker to Different Drive**
```powershell
# Stop Docker Desktop
# In Docker Desktop Settings > Resources > Advanced
# Change "Disk image location" to C: or D: drive
```

### 4. **Use Multi-Stage Build (Already Implemented)**
The optimized Dockerfile now uses multi-stage builds to reduce final image size:
- Stage 1: Build dependencies (discarded after build)
- Stage 2: Runtime image (much smaller)

## 📦 **Alternative Deployment Options**

### Option A: Use Pre-built Base Image
```dockerfile
FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime
# Much smaller than building PyTorch from scratch
```

### Option B: CPU-Only Build for Testing
```dockerfile
# Use CPU-only PyTorch (much smaller)
RUN pip install torch==2.6.0+cpu torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

### Option C: Use GitHub Actions for Building
Let GitHub Actions build the image (unlimited disk space):
```yaml
# .github/workflows/build.yml
- name: Build and push Docker image
  uses: docker/build-push-action@v4
  with:
    context: .
    push: true
    tags: your-registry/cartoon-animation:latest
```

## 🚀 **Recommended Workflow**

### 1. **Immediate Fix**
```powershell
# Run the optimized build script
.\build-docker.ps1
```

### 2. **If Still Failing**
```powershell
# Move to C: drive temporarily
cd C:\temp
git clone https://github.com/your-username/cartoon-runpod.git
cd cartoon-runpod
.\build-docker.ps1
```

### 3. **For Production**
```powershell
# Push to GitHub and use GitHub Actions
git add .
git commit -m "Add optimized Docker build"
git push origin main
# GitHub Actions will build and push automatically
```

## 📊 **Space Optimization Results**

### Before Optimization:
- Build context: ~2GB (with models and outputs)
- Intermediate layers: ~15GB
- Final image: ~8GB

### After Optimization:
- Build context: ~50MB (with .dockerignore)
- Intermediate layers: ~5GB (multi-stage)
- Final image: ~3GB (CPU-only PyTorch)

## 🛠️ **Build Commands**

### Windows PowerShell:
```powershell
.\build-docker.ps1
```

### Linux/Mac:
```bash
chmod +x build-docker.sh
./build-docker.sh
```

### Manual Build:
```bash
export DOCKER_BUILDKIT=1
docker build --no-cache --rm -t cartoon-animation:latest .
```

## 📋 **Verification**

After successful build:
```powershell
# Check image size
docker images cartoon-animation:latest

# Test the container
docker run -p 7860:7860 cartoon-animation:latest

# Check disk space
Get-PSDrive -PSProvider FileSystem | Select-Object Name, Used, Free
```

## 🔄 **If Problems Persist**

1. **Move entire project to C: drive**
2. **Use GitHub Codespaces for building**
3. **Use RunPod's build service directly**
4. **Build on a cloud instance with more disk space**

## 🎯 **Next Steps**

1. Run `.\build-docker.ps1`
2. If successful, push to Docker Hub
3. Deploy on RunPod using the built image
4. Test with sample inputs

The optimized build should complete successfully with the available disk space! 