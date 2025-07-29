# QueerGen: Evaluating Representational Harm Toward Queer Identities in Language Models

**QueerGen** is a research framework designed to evaluate how language models represent queer and non-normative identities through text generation. It focuses on analyzing **sentiment**, **regard**, **toxicity**, and **lexical diversity** across various subject categories — *unmarked*, *non-queer*, and *queer* — using controlled prompt templates and a suite of evaluation tools.

---
---

## 📚 Table of Contents

- [📁 Project Structure](#-project-structure)
- [⚙️ Installation](#️-installation)
  - [Set up the environment](#set-up-the-environment)
- [📄 Dataset Generation](#-dataset-generation)
- [🧠 Sentence Completion](#-sentence-completion)
  - [🔐 API Keys](#-api-keys)
  - [🧩 Model Installation](#-model-installation)
  - [Sentence Completion](#sentence-completion)
- [📈 Completions Evaluation](#-completions-evaluation)
  - [🔐 API Key](#-api-key)
  - [📦 Install Evaluation Tools](#-install-evaluation-tools)
  - [Evaluation Completions](#evaluation-completions)
- [📊 Generate Graphs](#-generate-graphs)
- [📌 Summary of `env_config.sh` Options](#-summary-of-env_configsh-options)

---

## 📌 Summary of `env_config.sh` Options

| Flag         | Description                                                                 |
|--------------|-----------------------------------------------------------------------------|
| `--setup`    | Creates a Python virtual environment and installs core dependencies        |
| `--models`   | Installs API libraries and pulls local language models via Ollama          |
| `--evaluate` | Installs sentiment, regard, and toxicity evaluation tools and dependencies |

Example usage:
```bash
bash env_config.sh --setup        # Set up the Python environment
bash env_config.sh --models       # Install model libraries and local Ollama models
bash env_config.sh --evaluate     # Install tools for evaluating sentence completions
```

## 📁 Project Structure
```
.
├── dataset/                         # Folder containing input datasets and template components
│   ├── template_complete.csv              # Final dataset of all generated prompt templates
│   ├── markers.csv                        # CSV file containing the queer and non-queer markers
│   ├── subjects.csv                       # CSV file containing subject terms (e.g. person identities)
│   ├── templates.csv                      # CSV file with sentence template structures for prompt generation
│
├── generations/                      # Model sentence completions are saved here (sub-folder divided by prompt type)
│
├── evaluations/                     # Evaluation scores (sentiment, regard, toxicity) are stored here (sub-folder divided by prompt type)
│
├── graphs/                         # Visualizations and comparative analysis plots are saved here
|
├── tables/                         # Tables and data analysis are saved here
│
├── env_config.sh                          # Environment setup script to install dependencies and models
│
├── ├── src/                         # Folder containing the scripts to:
│   ├── evaluation.py               Evaluates model outputs on sentiment, regard, and toxicity
│   ├── lib.py                       # Shared utility contants used across other Python scripts
|   ├── models.py                           # Runs sentence completion using  language models
│   ├── template.py                     # Script that generates template_complete.csv from CSV components 
│                 
│
├── graphs.ipynb                 # Evaluates model outputs on sentiment, regard, and toxicity
│
├── main.py                      # main 
│
├── lib.py                                 # Shared utility functions used across other Python scripts
│
└── .env                                   # Environment variables for API keys (OpenAI, Perspective, etc.)
```
## ⚙️ Installation
### Set up the environment

Use the `env_config.sh` script with flags to install dependencies and configure your environment:
```bash
bash env_config.sh --setup
```

## 📄 Dataset generation
To generate the base dataset (template_complete.csv) used for sentence generation:
```bash
python3 createTemplate.py 
```

This will create the file in the `dataset_source/` directory.

## 🧠 Sentence completion
To run the sentence completion task, install the necessary models and libraries.

### 🔐 API Keys
Add your API keys to a .env file in the root directory:
- DeepSeek 671b parameters model: Requires an API key. Add it to the ```.env``` file to the DEEPSEEK_API_KEY variable (it is also possible to accessi it through Ollama API for free - we opted to use this API as it is less time consuming)
- OpenAI GPT-4 Models: Requires an API key. Add it to the ```.env``` file to the OPENAI_API_KEY variable
- GenAI Gemini 2.0 Flash Models: Requires an API key. Add it to the ```.env``` file to the GENAI_API_KEY variable

Resulting in the following `.env` file:
generation:
```env
DEEPSEEK_API_KEY=your_deepseek_key
OPENAI_API_KEY=your_openai_key
GENAI_API_KEY=your_genai_key
```

### 🧩 Model Installation
To generate predictions with the considered models it is necessary to install the corrispondent libraries and local models. This is a list of the required library, models and framework:
```bash
bash env_config.sh --models
```

This installs:
- Ollama framework for building and running language models on the local machine. And the following models through Ollama service:
    - [Llama 3](https://ollama.com/library/llama3)  8b (4.7GB) and 70b parameters (40GB) 
    - [Gemma 3](https://ollama.com/library/gemma3) 4b parameters (3.3GB) and 27b parameters (17GB)
    - [DeepSeek R1](https://ollama.com/library/deepseek-r1) 8b parameters 5.2GB  
- OpenAI library for:
    - GPT-4o
    - GPT-4o mini 
- GenAI library for:
    - Gemini 2.0 Flash
    - Gemini 2.0 Flash Light 
- DeepSeek library for:
    - DeepSeek R1 671b parameters

Note: Ollama models are downloaded locally. Ensure you have sufficient disk space (models range from ~3GB to ~40GB).

### Sentence completion
To generate the predictions and perform sentence completions:
```bash
python3 generateSentences.py 
```

Generated completions will be saved in the `output_sentences/` folder.

## 📈 Completions Evaluation
To evaluate completions is necessary to install the libraries to perform sentiment analysis, regard scores and toxicity classification.

### 🔐 API Key
Add your Perspective API key to the `.env` file:
```env
PERSPECTIVE_API_KEY=your_perspective_key
```
Check [Perpective API website](https://developers.perspectiveapi.com/s/docs?language=en_US) for more instructions.

### 📦 Install Evaluation Tools
Run:
```bash
bash env_config.sh --evaluate
```
This installs:
- [VADER](https://github.com/cjhutto/vaderSentiment) (Valence Aware Dictionary and sEntiment Reasoner) for sentiment analysis.
- [Hugging Face’s Evaluate library](https://github.com/huggingface/evaluate) (cloned manually) for computing regard scores. This will create the folder `evaluate` in `venv/evaluate` and enables to calculate the regard score.

### Evaluation completions
To generate the predictions and perform sentence completions:
```bash
python3 evaluateCompletions.py 
```
The completions will be available in the folder `output_evaluation/`. 

## 📊 Generate Graphs
To generate the graphs resulting from the evaluations:

```bash
python3 generateGraphs.py 
```
Graphs will be saved in the `output_graphs/` folder.