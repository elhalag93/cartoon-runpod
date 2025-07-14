#!/bin/bash

# Cartoon Animation Docker Build Script - Space Optimized
# This script builds the Docker image with space optimization techniques

set -e

echo "🚀 Starting Cartoon Animation Docker Build (Space Optimized)"
echo "============================================================"

# Check available disk space
echo "📊 Checking disk space..."
df -h

# Clean up Docker to free space
echo "🧹 Cleaning up Docker cache..."
docker system prune -f --volumes || echo "⚠️ Docker cleanup failed or not needed"

# Set build arguments for optimization
BUILD_ARGS="--no-cache --rm"
IMAGE_NAME="cartoon-animation"
TAG="latest"

# Build with BuildKit for better caching and smaller layers
echo "🔨 Building Docker image with BuildKit..."
export DOCKER_BUILDKIT=1

# Build the image
docker build $BUILD_ARGS -t $IMAGE_NAME:$TAG . || {
    echo "❌ Build failed. Trying with additional cleanup..."
    
    # Additional cleanup
    docker image prune -f
    docker container prune -f
    
    # Try building again with more aggressive cleanup
    echo "🔄 Retrying build with aggressive cleanup..."
    docker build --no-cache --rm --force-rm -t $IMAGE_NAME:$TAG .
}

# Clean up intermediate images
echo "🧹 Cleaning up intermediate images..."
docker image prune -f

# Show final image size
echo "📏 Final image size:"
docker images $IMAGE_NAME:$TAG

echo "✅ Build completed successfully!"
echo "🐳 Image: $IMAGE_NAME:$TAG"
echo ""
echo "🚀 To run the container:"
echo "   docker run -p 7860:7860 --gpus all $IMAGE_NAME:$TAG"
echo ""
echo "📤 To push to registry:"
echo "   docker tag $IMAGE_NAME:$TAG your-registry/$IMAGE_NAME:$TAG"
echo "   docker push your-registry/$IMAGE_NAME:$TAG" 