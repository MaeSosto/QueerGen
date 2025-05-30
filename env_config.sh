#!/bin/bash
set -e

function setup_environment() {
    echo "🐍 Creating Python virtual environment..."
    python3 -m venv .venv
    source .venv/bin/activate

    echo "📦 Upgrading pip..."
    pip install --upgrade pip

    echo "🔧 Installing base libraries..."
    for pkg in torch pandas tqdm unidecode surprisal transformers python-dotenv; do
        echo "   → Installing $pkg..."
        pip install "$pkg" || { echo "❌ Failed to install $pkg"; exit 1; }
    done

    echo "📊 Installing graph & ML libraries..."
    for pkg in tf-keras seaborn scikit-learn scipy matplotlib wordcloud python-ternary; do
        echo "   → Installing $pkg..."
        pip install "$pkg" || { echo "❌ Failed to install $pkg"; exit 1; }
    done

    echo "✅ Environment setup complete!"
}

function install_models() {
    echo "🤖 Installing Closed-model APIs..."

    for pkg in openai google-generativeai; do
        echo "   → Installing $pkg..."
        pip install "$pkg" || { echo "❌ Failed to install $pkg"; exit 1; }
    done

    echo "📦 Installing Ollama (local model runner)..."
    pip install ollama || { echo "❌ Failed to install ollama"; exit 1; }

    echo "📥 Pulling local LLMs with Ollama:"
    
    MODELS=("llama3" "llama3:70b" "gemma3" "gemma3:27b" "deepseek-r1")
    for model in "${MODELS[@]}"; do
        echo "   → Pulling $model..."
        ollama pull "$model" || { echo "❌ Failed to pull $model"; exit 1; }
    done

    echo "🚀 Starting Ollama server in background..."
    ollama serve > /dev/null &
}

function evaluate_sentences() {
    echo "🧠 Installing sentiment analysis tools..."

    for pkg in vadersentiment; do
        echo "   → Installing $pkg..."
        pip install "$pkg" || { echo "❌ Failed to install $pkg"; exit 1; }
    done

    echo "📁 Installing Hugging Face evaluation tools (local clone)..."
    cd .venv || { echo "❌ Failed to enter .venv directory"; exit 1; }

    if [ ! -d "evaluate" ]; then
        git clone https://github.com/huggingface/evaluate.git || { echo "❌ Git clone failed"; exit 1; }
    else
        echo "   → 'evaluate' already cloned, skipping..."
    fi

    pip install evaluate || { echo "❌ Failed to install evaluate"; exit 1; }

    cd ..
}

if [ $# -eq 0 ]; then
    echo "❗ Usage: $0 [--setup] [--models] [--evaluate]"
    exit 1
fi

source .venv/bin/activate 2> /dev/null || true

for arg in "$@"
do
    case $arg in
        --setup)
            setup_environment
            ;;
        --models)
            install_models
            ;;
        --evaluate)
            evaluate_sentences
            ;;
        *)
            echo "❗ Unknown option: $arg"
            echo "   Use: $0 [--setup] [--models] [--evaluate]"
            exit 1
            ;;
    esac
done