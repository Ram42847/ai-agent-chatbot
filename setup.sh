#!/bin/bash

echo "======================================"
echo "AI Agent Chatbot - Setup Script"
echo "======================================"

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check Python version
echo -e "\n${YELLOW}[1/7] Checking Python version...${NC}"
python_version=$(python3 --version 2>&1 | awk '{print $2}')
required_version="3.9"

if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" = "$required_version" ]; then 
    echo -e "${GREEN}✓ Python $python_version installed${NC}"
else
    echo "✗ Python 3.9+ required. Found: $python_version"
    exit 1
fi

# Create virtual environment
echo -e "\n${YELLOW}[2/7] Creating virtual environment...${NC}"
python3 -m venv venv
echo -e "${GREEN}✓ Virtual environment created${NC}"

# Activate virtual environment
echo -e "\n${YELLOW}[3/7] Activating virtual environment...${NC}"
source venv/bin/activate
echo -e "${GREEN}✓ Virtual environment activated${NC}"

# Install dependencies
echo -e "\n${YELLOW}[4/7] Installing Python dependencies...${NC}"
pip install --upgrade pip
pip install -r requirements.txt
echo -e "${GREEN}✓ Dependencies installed${NC}"

# Check Ollama installation
echo -e "\n${YELLOW}[5/7] Checking Ollama installation...${NC}"
if command -v ollama &> /dev/null; then
    echo -e "${GREEN}✓ Ollama is installed${NC}"
else
    echo -e "${YELLOW}⚠ Ollama not found. Installing...${NC}"
    curl -fsSL https://ollama.ai/install.sh | sh
fi

# Start Ollama service
echo -e "\n${YELLOW}[6/7] Starting Ollama service...${NC}"
ollama serve &
sleep 5
echo -e "${GREEN}✓ Ollama service started${NC}"

# Download models
echo -e "\n${YELLOW}[7/7] Downloading LLM models...${NC}"
echo "Downloading Llama 3.1 (8B) - This may take a few minutes..."
ollama pull llama3.1:8b
echo -e "${GREEN}✓ Models downloaded${NC}"

# Create .env file if not exists
if [ ! -f .env ]; then
    echo -e "\n${YELLOW}Creating .env file...${NC}"
    cp .env.example .env
    echo -e "${GREEN}✓ .env file created${NC}"
    echo -e "${YELLOW}⚠ Please update .env with your API keys if needed${NC}"
fi

# Create required directories
mkdir -p data/user_uploads
mkdir -p logs

echo -e "\n======================================"
echo -e "${GREEN}✅ Setup Complete!${NC}"
echo -e "======================================\n"
echo "To run the application:"
echo "  1. Activate virtual environment: source venv/bin/activate"
echo "  2. Run: python main.py"
echo ""
echo "The application will be available at: http://localhost:7860"
echo ""

#Your new public key is: 

#ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIJgxrvYPD0FTvNeqLUM/Zj9CkiWMtTIywjQPHrvIdxhQ"""