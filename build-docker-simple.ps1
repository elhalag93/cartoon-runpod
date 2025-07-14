# Cartoon Animation Docker Build Script - Windows PowerShell (Simple Version)
# This script builds the Docker image with space optimization techniques

$ErrorActionPreference = "Stop"

Write-Host "Starting Cartoon Animation Docker Build (Space Optimized)" -ForegroundColor Green
Write-Host "==========================================================" -ForegroundColor Green

# Check available disk space
Write-Host "Checking disk space..." -ForegroundColor Yellow
Get-PSDrive -PSProvider FileSystem | Select-Object Name, Used, Free | Format-Table

# Clean up Docker to free space
Write-Host "Cleaning up Docker cache..." -ForegroundColor Yellow
try {
    docker system prune -f --volumes
    Write-Host "Docker cleanup completed" -ForegroundColor Green
} catch {
    Write-Host "Docker cleanup failed or not needed" -ForegroundColor Yellow
}

# Set build parameters
$IMAGE_NAME = "cartoon-animation"
$TAG = "latest"
$BUILD_ARGS = "--no-cache --rm"

# Enable BuildKit for better caching
$env:DOCKER_BUILDKIT = "1"

Write-Host "Building Docker image with BuildKit..." -ForegroundColor Yellow

try {
    # Build the image
    Invoke-Expression "docker build $BUILD_ARGS -t ${IMAGE_NAME}:${TAG} ."
    Write-Host "Build completed successfully!" -ForegroundColor Green
} catch {
    Write-Host "Build failed. Trying with additional cleanup..." -ForegroundColor Red
    
    # Additional cleanup
    docker image prune -f
    docker container prune -f
    
    # Try building again with more aggressive cleanup
    Write-Host "Retrying build with aggressive cleanup..." -ForegroundColor Yellow
    Invoke-Expression "docker build --no-cache --rm --force-rm -t ${IMAGE_NAME}:${TAG} ."
}

# Clean up intermediate images
Write-Host "Cleaning up intermediate images..." -ForegroundColor Yellow
docker image prune -f

# Show final image size
Write-Host "Final image size:" -ForegroundColor Yellow
docker images "${IMAGE_NAME}:${TAG}"

Write-Host ""
Write-Host "Image: ${IMAGE_NAME}:${TAG}" -ForegroundColor Green
Write-Host ""
Write-Host "To run the container:" -ForegroundColor Cyan
Write-Host "   docker run -p 7860:7860 --gpus all ${IMAGE_NAME}:${TAG}" -ForegroundColor White
Write-Host ""
Write-Host "To push to registry:" -ForegroundColor Cyan
Write-Host "   docker tag ${IMAGE_NAME}:${TAG} your-registry/${IMAGE_NAME}:${TAG}" -ForegroundColor White
Write-Host "   docker push your-registry/${IMAGE_NAME}:${TAG}" -ForegroundColor White

# Final disk space check
Write-Host ""
Write-Host "Final disk space:" -ForegroundColor Yellow
Get-PSDrive -PSProvider FileSystem | Select-Object Name, Used, Free | Format-Table 