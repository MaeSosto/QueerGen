#!/bin/bash

# Exit immediately on error
set -e

echo "🐍 Creating Python virtual environment..."
python3 -m venv .venv > /dev/null
source .venv/bin/activate > /dev/null

echo "📦 Upgrading pip..."
pip install --upgrade pip > /dev/null

echo "📁 Cloning Hugging Face evaluation tools..."
cd .venv
git clone https://github.com/huggingface/evaluate.git > /dev/null
pip install evaluate > /dev/null
cd ..

echo "📦 Installing Ollama and starting the service..."
pip install ollama > /dev/null
# Uncomment the models you want to pull
# ollama pull llama3
# ollama pull llama3:70b
# ollama pull gemma3
# ollama pull gemma3:27b
# ollama pull deepseek-r1
# ollama pull deepseek-r1:70b
ollama serve > /dev/null &

echo "🔧 Installing base libraries..."
pip install torch pandas tqdm unidecode surprisal transformers python-dotenv > /dev/null

echo "🧠 Installing sentiment analysis tools..."
pip install afinn vadersentiment > /dev/null

echo "☁️ Setting up Google Cloud clients..."
pip install --upgrade google-api-python-client google-cloud google-cloud-language google-cloud-translate > /dev/null

echo "🤖 Installing model APIs..."
pip install openai google-generativeai > /dev/null

echo "🔧 Installing ipykernel for Jupyter support..."
pip install ipykernel > /dev/null

echo "📊 Installing graph & ML libraries..."
pip install tf-keras seaborn scikit-learn scipy matplotlib wordcloud python-ternary > /dev/null

echo "✅ Environment setup complete!"