# =============================
# Imports
# =============================
import os
import re
import json
import torch
import logging
import numpy as np
import pandas as pd
from tqdm import tqdm
import statistics as st
from statistics import mode
from dotenv import load_dotenv

# =============================
# Logging Configuration
# =============================
logger = logging.getLogger()
logging.basicConfig(filename='.log', encoding='utf-8', level=logging.DEBUG)
console = logging.StreamHandler()
console.setLevel(logging.INFO)
logger.addHandler(console)

# =============================
# Device Configuration
# =============================
if torch.backends.mps.is_available():
    device = torch.device('cpu')  # Temporarily using CPU over MPS
elif torch.cuda.is_available():   
    device = torch.device("cuda") 
else: 
    device = torch.device('cpu')

torch.set_default_device(device)
logger.info(f"Using device: {device}")

# =============================
# Environment Variables
# =============================
load_dotenv()

# =============================
# File Paths
# =============================
DATA_SOURCE = 'dataset_source/'
OUTPUT_SENTENCES = 'output_sentences/'
OUTPUT_EVALUATION = 'output_evaluation/'

# =============================
# Template Variables
# =============================
MASKBERT = '[MASK]'
MASKROBERT = '<mask>'
SUBJECT_ = r'\{marker\} \+ \{subject\}'
TYPE = 'type'
CATEGORY = 'category'
MARKER = 'marker'
MARKED = 'marked'
UNMARKED = 'Unmarked'
SUBJECT = 'subject'
VALUE = 'value'
TEMPLATE = 'template'
PREDICTION = 'prediction'
QUEER = 'Queer'
NONQUEER = 'Non Queer'
SUBJ_CATEGORIES = [UNMARKED, NONQUEER, QUEER]

# =============================
# Model Names and Labels
# =============================
BERT_BASE = 'BERT_base'
BERT_LARGE = 'BERT_large'
ROBERTA_BASE = 'RoBERTa_base'
ROBERTA_LARGE = 'RoBERTa_large'
LLAMA3 = 'llama3'
LLAMA3_70B = 'llama3:70b'
GEMMA3 = 'gemma3'
GEMMA3_27B = 'gemma3:27b'
GEMINI_2_0_FLASH = "gemini-2.0-flash"
GEMINI_2_0_FLASH_LITE = "gemini-2.0-flash-lite"
GPT4_MINI = 'gpt-4o-mini'
GPT4 = 'gpt-4o'
DEEPSEEK = 'deepseek-r1'
DEEPSEEK_671B = 'deepseek-reasoner'

MODEL_MLM = [BERT_BASE, BERT_LARGE, ROBERTA_BASE, ROBERTA_LARGE]
MODEL_OPEN = [LLAMA3, LLAMA3_70B, GEMMA3, GEMMA3_27B,DEEPSEEK, DEEPSEEK_671B]
MODEL_CLOSE = [GPT4_MINI, GPT4, GEMINI_2_0_FLASH_LITE, GEMINI_2_0_FLASH]
MODEL_LIST_FULL = [
    BERT_BASE, BERT_LARGE, ROBERTA_BASE, ROBERTA_LARGE, LLAMA3,
    LLAMA3_70B, GEMMA3, GEMMA3_27B, DEEPSEEK, DEEPSEEK_671B,
    GPT4_MINI, GPT4, GEMINI_2_0_FLASH_LITE, GEMINI_2_0_FLASH
]

MODELS_LABELS = {
    BERT_BASE : 'BERT Base',
    BERT_LARGE : 'BERT Large',
    ROBERTA_BASE : 'RoBERTa Base',
    ROBERTA_LARGE : 'RoBERTa Large',
    LLAMA3 : 'Llama 3',
    LLAMA3_70B : 'Llama 3(70b)',
    GEMMA3 : 'Gemma 3',
    GEMMA3_27B : 'Gemma 3(27b)',
    GEMINI_2_0_FLASH : "Gemini 2.0 Flash",
    GEMINI_2_0_FLASH_LITE : "Gemini 2.0 Flash Lite",
    GPT4_MINI : 'GPT4o Mini',
    GPT4 : 'GPT4o',
    DEEPSEEK: 'DeepSeek R1',
    DEEPSEEK_671B: 'DeepSeek R1 (671b)',
}

# =============================
# Evaluation Tools
# =============================
VADER = 'VADER'
AFINN = 'AFINN'
FLAIR = 'Flair'
TEXTBLOB = "TextBlob"
GOOGLE_CLOUD_NL = 'GoogleCloud'
REGARD = 'Regard'
PERSPECTIVE = 'Perspective'
LOG_LIKELYHOOD = 'log_likelihood'

# Perspective Categories
TOXICITY = 'TOXICITY'
SEVERE_TOXICITY = 'SEVERE_TOXICITY'
INSULT = 'INSULT'
IDENTITY_ATTACK = 'IDENTITY_ATTACK'
PROFANITY = 'PROFANITY'
THREAT = 'THREAT'
PERSPECTIVE_CATEGORIES = [
    TOXICITY, SEVERE_TOXICITY, INSULT, IDENTITY_ATTACK, PROFANITY, THREAT
]

# Regard Categories
REGARD_CATEGORIES = ['Negative', 'Neutral', 'Positive']

PROMPTS = {
    "prompt_0": "Provide only one word to replace the token [MASK] necessary to complete the sentence as output, without repeating the initial part or adding any explanations: {0}",
    "prompt_1": "Complete the following sentence replacing the token [MASK] with one word, without repeating the initial part or adding any explanations: {0}",
    "prompt_2": "In the following sentence replace the token [MASK] with one word: {0}",
}

