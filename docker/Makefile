.PHONY: help build run stop clean logs shell

help:
	@echo "AI Agent Chatbot - Docker Commands"
	@echo "=================================="
	@echo "make build          - Build Docker images"
	@echo "make run            - Run with docker-compose (app + ollama)"
	@echo "make run-full       - Run all-in-one container"
	@echo "make run-gemini     - Run with Gemini only"
	@echo "make stop           - Stop all containers"
	@echo "make clean          - Remove containers and volumes"
	@echo "make logs           - View application logs"
	@echo "make logs-ollama    - View Ollama logs"
	@echo "make shell          - Open shell in app container"
	@echo "make pull-models    - Pull Ollama models"

build:
	docker-compose build

run:
	docker-compose up -d
	@echo "Application running at http://localhost:7860"

run-full:
	docker-compose -f docker-compose.full.yml up -d
	@echo "Application running at http://localhost:7860"

run-gemini:
	docker-compose -f docker-compose.gemini.yml up -d
	@echo "Application running at http://localhost:7860"

stop:
	docker-compose down
	docker-compose -f docker-compose.full.yml down
	docker-compose -f docker-compose.gemini.yml down

clean:
	docker-compose down -v
	docker-compose -f docker-compose.full.yml down -v
	docker-compose -f docker-compose.gemini.yml down -v
	docker system prune -f

logs:
	docker-compose logs -f ai-agent-app

logs-ollama:
	docker-compose logs -f ollama

logs-full:
	docker-compose -f docker-compose.full.yml logs -f

shell:
	docker exec -it ai-agent-chatbot /bin/bash

pull-models:
	docker exec -it ollama-service ollama pull llama3.1:8b

ps:
	docker-compose ps
