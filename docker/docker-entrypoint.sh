#!/bin/bash
set -e

echo "Starting Ollama service..."
ollama serve &
OLLAMA_PID=$!

# Wait for Ollama to be ready
echo "Waiting for Ollama to start..."
until curl -s http://localhost:11434/api/version > /dev/null 2>&1; do
    sleep 2
done

echo "Ollama is ready!"

# Pull required models if not already present
echo "Checking for required models..."
if ! ollama list | grep -q "llama3.1:8b"; then
    echo "Pulling llama3.1:8b model..."
    ollama pull llama3.1:8b
fi

echo "Starting AI Agent application..."
python hybrid_main.py

# Keep script running and forward signals
wait $OLLAMA_PID
