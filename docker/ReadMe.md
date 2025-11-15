# ============================================================================
# USAGE INSTRUCTIONS
# ============================================================================

"""
DOCKER DEPLOYMENT OPTIONS
==========================

Option 1: App + Separate Ollama (Recommended)
----------------------------------------------
docker-compose up -d

Pros: 
- Ollama can be shared across projects
- Easier to manage
- Better resource isolation

Option 2: All-in-One Container
------------------------------
docker-compose -f docker-compose.full.yml up -d

Pros:
- Single container
- Simpler deployment
- Everything bundled

Option 3: Gemini Only (Lightweight)
-----------------------------------
docker-compose -f docker-compose.gemini.yml up -d

Pros:
- Smallest image
- No local models
- Fast startup

BUILD COMMANDS
==============

# Build base image
docker build -t ai-agent-chatbot:latest .

# Build with Ollama
docker build -t ai-agent-chatbot:ollama -f Dockerfile.ollama .

# Build for specific platform
docker build --platform linux/amd64 -t ai-agent-chatbot:latest .

RUN COMMANDS
============

# Run with environment file
docker run -d \
  --name ai-agent \
  -p 7860:7860 \
  --env-file .env \
  -v $(pwd)/chroma_db:/app/chroma_db \
  ai-agent-chatbot:latest

# Run with GPU support
docker run -d \
  --name ai-agent \
  --gpus all \
  -p 7860:7860 \
  --env-file .env \
  ai-agent-chatbot:ollama

# Run interactive
docker run -it \
  -p 7860:7860 \
  --env-file .env \
  ai-agent-chatbot:latest \
  /bin/bash

DEVELOPMENT
===========

# Run with code mounted (hot reload)
docker run -d \
  -p 7860:7860 \
  -v $(pwd):/app \
  --env-file .env \
  ai-agent-chatbot:latest

# View logs
docker logs -f ai-agent

# Execute commands in container
docker exec -it ai-agent python -c "from main import AIAgent; print(AIAgent())"

DOCKER COMPOSE COMMANDS
=======================

# Start services
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down

# Rebuild and restart
docker-compose up -d --build

# Scale services (if needed)
docker-compose up -d --scale ai-agent-app=3

# Remove everything including volumes
docker-compose down -v

TROUBLESHOOTING
===============

# Container not starting
docker logs ai-agent-chatbot

# Check resource usage
docker stats ai-agent-chatbot

# Inspect container
docker inspect ai-agent-chatbot

# Access container shell
docker exec -it ai-agent-chatbot /bin/bash

# Test Ollama connectivity
docker exec -it ai-agent-chatbot curl http://ollama:11434/api/version

# Pull models manually
docker exec -it ollama-service ollama pull llama3.1:8b

PRODUCTION DEPLOYMENT
====================

# Use docker-compose with production settings
docker-compose -f docker-compose.yml -f docker-compose.prod.yml up -d

# With reverse proxy (nginx)
docker-compose -f docker-compose.yml -f docker-compose.nginx.yml up -d

# With monitoring
docker-compose -f docker-compose.yml -f docker-compose.monitoring.yml up -d
"""

print("Docker configuration files created successfully!")
print("\nQuick Start:")
print("1. docker-compose up -d")
print("2. Visit http://localhost:7860")
print("\nFor more options, run: make help")