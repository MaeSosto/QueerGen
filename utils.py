import torch
import logging
import requests
from tqdm import tqdm
import pandas as pd
import os
import time
import re
import google.generativeai as genai
from openai import OpenAI

logging.basicConfig(level=logging.INFO)# OPTIONAL
print(f"PyTorch version: {torch.__version__}")

# Set the device      
device = "mps" if torch.backends.mps.is_available() else torch.device("cuda") if torch.cuda.is_available() else torch.device('cpu')
print(f"Using device: {device}")

#Data Source
DATA_SOURCE = 'dataset_source/'
OUTPUT_TEMPLATE = 'output_template/'
TEMPLATE_PATH = DATA_SOURCE + 'template.csv'
NOUNS_PATH = DATA_SOURCE + 'nouns.csv'
TEMPLATES_COMPLETE_PATH = OUTPUT_TEMPLATE + 'template_complete.csv'

# TEMPLATE MAP
TARGET_ = '<target>'
BE_ = '<be>'
HAVE_ = '<have>'
WERE_ = '<were>'
TYPE = 'type'
CATEGORY= 'category'
SUBJECT = 'subject'
PERSON = 'person'
THE = 'the'
TEMPLATE = 'template'
GENERATED = 'generated'

#Data Source
OUTPUT_TEMPLATE = 'output_template/'
TEMPLATES_COMPLETE_PATH = OUTPUT_TEMPLATE + 'template_complete.csv'
DATA_SOURCE = 'dataset_source/'
OUTPUT_PREDICTION = 'output_prediction/'

#Ollama Models
URL_OLLAMA_LOCAL = "http://localhost:11434/api/generate"
LLAMA3 = 'llama3'
LLAMA3_3 = 'llama3.3'
GEMMA2 = 'gemma2'
GEMMA2_27B = 'gemma2:27b'
GEMINI_FLASH = "gemini-1.5-flash"
GPT4_MINI = 'gpt-4o-mini'
GPT4 = 'gpt-4o'
MODEL_LIST = [LLAMA3, LLAMA3_3, GEMMA2, GEMMA2_27B, GEMINI_FLASH, GPT4, GPT4_MINI]

#API KEY
GENAI_API_KEY = 'AIzaSyAk29hD3nisZTXfr14rK7rdn_pZ_dQ8jDI'
OPENAI_API_KEY = "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"
