import torch
import logging
import requests
from tqdm import tqdm
import pandas as pd
import os

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

#Data Source
OUTPUT_TEMPLATE = 'output_template/'
TEMPLATES_COMPLETE_PATH = OUTPUT_TEMPLATE + 'template_complete.csv'
DATA_SOURCE = 'dataset_source/'
OUTPUT_PREDICTION = 'output_prediction/'

#Ollama Models
url = "http://localhost:11434/api/generate"
LLAMA3 = 'llama3'
LLAMA3_3 = 'llama3.3'
GEMMA2 = 'gemma2'
GEMMA2_27B = 'gemma2:27b'
MODEL_LIST = [LLAMA3, LLAMA3_3, GEMMA2, GEMMA2_27B]