group "default" {
  targets = ["cartoon-worker"]
}

target "cartoon-worker" {
  dockerfile = "Dockerfile"
  context = "."
  tags = ["cartoon-animation-worker:latest"]
  platforms = ["linux/amd64"]
  
  args = {
    BUILDKIT_INLINE_CACHE = 1
  }
  
  cache-from = [
    "type=gha"
  ]
  
  cache-to = [
    "type=gha,mode=max"
  ]
} 