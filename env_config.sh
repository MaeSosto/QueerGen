#!/bin/bash
echo "🐍 Creating Python virtual environment..."
python3 -m venv .venv
source .venv/bin/activate

echo "📦 Upgrading pip..."
pip install --upgrade pip

echo "🔧 Installing base libraries..."
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
for pkg in pandas tqdm unidecode surprisal transformers python-dotenv; do
    echo "   → Installing $pkg..."
    pip install "$pkg" || { echo "❌ Failed to install $pkg"; exit 1; }
done

echo "📊 Installing graph & ML libraries..."
for pkg in tf-keras seaborn scikit-learn scipy matplotlib wordcloud python-ternary jupyter; do
    echo "   → Installing $pkg..."
    pip install "$pkg" || { echo "❌ Failed to install $pkg"; exit 1; }
    pip install ipywidgets --upgrade
    jupyter nbextension enable --py widgetsnbextension
done

echo "✅ Environment setup complete!"
echo "🤖 Installing Closed-model APIs..."

for pkg in openai google-generativeai; do
    echo "   → Installing $pkg..."
    pip install "$pkg" || { echo "❌ Failed to install $pkg"; exit 1; }
done

echo "📦 Installing Ollama (local model runner)..."
pip install ollama || { echo "❌ Failed to install ollama"; exit 1; }

echo "📥 Pulling local LLMs with Ollama:"

# MODELS=("llama3" "llama3:70b" "gemma3" "gemma3:27b" "deepseek-r1")
# for model in "${MODELS[@]}"; do
#     echo "   → Pulling $model..."
#     ollama pull "$model" || { echo "❌ Failed to pull $model"; exit 1; }
# done

echo "🚀 Starting Ollama server in background..."
ollama serve > /dev/null &

echo "🧠 Installing sentiment analysis tools..."
cd .venv
git clone https://github.com/fnielsen/afinn
cd afinn
python setup.py install
cd .. 
cd ..

for pkg in vadersentiment flair textblob spacy; do
    python -m textblob.download_corpora
    python -m spacy download en_core_web_sm
    python -m spacy download en
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
