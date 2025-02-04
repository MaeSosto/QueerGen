import torch
import logging
import requests
from tqdm import tqdm
import pandas as pd
import os
import re
import time
import sys
from openai import OpenAI
from googleapiclient import discovery
from collections import defaultdict
from evaluate import load 

logging.basicConfig(level=logging.INFO)# OPTIONAL
print(f"PyTorch version: {torch.__version__}")

# Set the device      
device = "mps" if torch.backends.mps.is_available() else torch.device("cuda") if torch.cuda.is_available() else torch.device('cpu')
print(f"Using device: {device}")

credential_path = "ADC.json"
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = credential_path

#Generate sentences
MASKBERT_ = r'\[MASK\]'

#Data Source
DATA_SOURCE = 'dataset_source/'
OUTPUT_SENTENCES = 'output_sentences/'
OUTPUT_EVALUATION = 'output_evaluation/'
TEMPLATE_PATH = DATA_SOURCE + 'template.csv'
NOUNS_PATH = DATA_SOURCE + 'nouns.csv'

#TYPE 
QUEER = 'queer'
NON_QUEER = 'non-queer'
NEUTRAL = 'neutral'
SUBJECT_TYPES = [QUEER, NON_QUEER, NEUTRAL]

# TEMPLATE MAP
TARGET_ = '<target>'
TYPE = 'type'
CATEGORY= 'category'
SUBJECT = 'subject'
PERSON = 'person'
THE = 'the'
TEMPLATE = 'template'
GENERATED = 'generated'

#Perspective
# TOXICITY = 'TOXICITY'
# INSULT = 'INSULT'
# IDENTITY_ATTACK = 'IDENTITY_ATTACK'
# PROFANITY = 'PROFANITY'
# THREAT = 'THREAT'
# PERSPECTIVE_CATEGORIES =[TOXICITY, INSULT, IDENTITY_ATTACK, PROFANITY, THREAT]

#Evaluation tools
VADER = 'VADER'
#PERSPECTIVE = 'Perspective'
GOOGLE_CLOUD_NL = 'GoogleCloud'
TEXTBLOB = 'textBlob'
AFINN = 'afinn'
PERPLEXITY = 'perplexity'
EVALUATION_TOOLS = [VADER, GOOGLE_CLOUD_NL, TEXTBLOB, AFINN, PERPLEXITY]

#Ollama Models
URL_OLLAMA_LOCAL = "http://localhost:11434/api/generate"
LLAMA3 = 'llama3'
LLAMA3_70B = 'llama3:70b'
LLAMA3_3 = 'llama3.3'
GEMMA2 = 'gemma2'
GEMMA2_27B = 'gemma2:27b'
GEMINI_FLASH = "gemini-1.5-flash"
GPT4_MINI = 'gpt-4o-mini'
GPT4 = 'gpt-4o'
MODEL_LIST = [LLAMA3, LLAMA3_70B, LLAMA3_3, GEMMA2, GEMMA2_27B, GEMINI_FLASH, GPT4, GPT4_MINI]