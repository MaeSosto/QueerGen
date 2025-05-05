#Imports
import torch
import logging
import pandas as pd
from tqdm import tqdm
from googleapiclient import discovery
import unidecode
import os
import requests
import time
import re
from collections import defaultdict
import json
import numpy as np
import statistics as st
from dotenv import load_dotenv

logger = logging.getLogger()
logging.basicConfig(filename='.log', encoding='utf-8', level=logging.DEBUG)
console = logging.StreamHandler()
console.setLevel(logging.INFO)
logger.addHandler(console)
#print(f"PyTorch version: {torch.__version__}")

# Set the device      
if torch.backends.mps.is_available():
    #device = torch.device("mps") 
    device = torch.device('cpu') 
elif torch.cuda.is_available():   
    device = torch.device("cuda") 
else: 
    device = torch.device('cpu')
torch.set_default_device(device)
logger.info(f"Using device: {device}")

#Get local evirnoment 
load_dotenv()

# # Global Variables
#Data Source
DATA_SOURCE = 'dataset_source/'
OUTPUT_TEMPLATE = 'output_template/'
OUTPUT_QUEERBENCH = 'output_queerbench/'
OUTPUT_GRAPHS = 'output_graphs/'
OUTPUT_PREDICTION = 'output_prediction/'
OUTPUT_SENTENCES = 'output_sentences/'
OUTPUT_EVALUATION = 'output_evaluation/'
EVALUATION_MEASUREMENT_PATH = '.venv/evaluate/measurements/'
EVALUATION_METRICS_PATH = '.venv/evaluate/metrics/'

# TEMPLATE MAP
BE_ = '<be>'
SUBJECT_ = r"\[subject\]"
HAVE_ = '<have>'
WERE_ = '<were>'
QUEERNESS = 'queerness'
TYPE = 'type'
CATEGORY= 'category'
MARKER = 'marker'
MARKED = 'marked'
UNMARKED = 'Unmarked'
SUBJECT = 'subject'
VALUE = 'value'
MASKBERT_ = r'\[MASK\]'
MASKBERT = '[MASK]'
MASKROBERT = '<mask>'
THE = 'the'
ADJ = 'adj'
TEMPLATE = 'template'
PREDICTION = 'prediction'
ORIGINAL = 'original'

# MODELS
BERT_BASE = 'BERT_base'
BERT_LARGE = 'BERT_large'
ROBERTA_BASE = 'RoBERTa_base'
ROBERTA_LARGE = 'RoBERTa_large'
LLAMA3 = 'llama3'
LLAMA3_70B = 'llama3:70b'
LLAMA3_3 = 'llama3.3'
GEMMA3 = 'gemma3'
GEMMA3_27B = 'gemma3:27b'
GEMINI_1_5_FLASH = "gemini-1.5-flash"
GEMINI_2_0_FLASH = "gemini-2.0-flash"
GEMINI_2_0_FLASH_LITE = "gemini-2.0-flash-lite"
GPT4_MINI = 'gpt-4o-mini'
GPT4 = 'gpt-4o'
DEEPSEEK = 'deepseek-r1'
DEEPSEEK_673B = 'deepseek-reasoner'
MODEL_LIST_FULL = [BERT_BASE, BERT_LARGE, ROBERTA_BASE, ROBERTA_LARGE, LLAMA3, LLAMA3_70B, GEMMA3, GEMMA3_27B, DEEPSEEK, DEEPSEEK_673B,  GPT4_MINI, GPT4, GEMINI_2_0_FLASH_LITE, GEMINI_2_0_FLASH]
MODELS_OPEN = [BERT_BASE, BERT_LARGE, LLAMA3, LLAMA3_70B, GEMMA3, GEMMA3_27B, DEEPSEEK, DEEPSEEK_673B]
MODELS_OPEN_NOLAMA = [BERT_BASE, BERT_LARGE, GEMMA3, GEMMA3_27B, DEEPSEEK, DEEPSEEK_673B]
MODELS_CLOSE = [GPT4_MINI, GPT4, GEMINI_2_0_FLASH_LITE, GEMINI_2_0_FLASH]
MODELS_SMALL = [BERT_BASE, LLAMA3, GEMMA3, DEEPSEEK,  GPT4_MINI, GEMINI_2_0_FLASH_LITE]
MODELS_BIG = [BERT_LARGE, LLAMA3_70B, GEMMA3_27B, DEEPSEEK_673B, GPT4, GEMINI_2_0_FLASH]

MODELS_LABELS = {
    BERT_BASE : 'BERT Base',
    BERT_LARGE : 'BERT Large',
    ROBERTA_BASE : 'RoBERTa Base',
    ROBERTA_LARGE : 'RoBERTa Large',
    LLAMA3_3 : 'Llama 3.3',
    LLAMA3 : 'Llama 3',
    LLAMA3_70B : 'Llama 3(70b)',
    GEMMA3 : 'Gemma 3',
    GEMMA3_27B : 'Gemma 3(27b)',
    GEMINI_1_5_FLASH : "Gemini 1.5 Flash",
    GEMINI_2_0_FLASH : "Gemini 2.0 Flash",
    GEMINI_2_0_FLASH_LITE : "Gemini 2.0 Flash Lite",
    GPT4_MINI : 'GPT4o Mini',
    GPT4 : 'GPT4o',
    DEEPSEEK_673B: 'DeepSeek R1 (671b)',
    DEEPSEEK: 'DeepSeek R1'
}

# TEMPLATE MAP
CATEGORY = 'category'
NAME = 'name'
TYPE = 'type'
UNMARKED = 'Unmarked'
QUEER = 'Queer'
NONQUEER = 'Non Queer'
ORIENTATION = 'orientation'
IDENTITY = 'identity'
QUEERBENCH = 'QueerBench'
SUBJ_CATEGORIES = [UNMARKED, NONQUEER, QUEER]
MARKER_CATEGORIES = [ORIENTATION, IDENTITY]

#Evaluation Tools
VADER = 'VADER'
AFINN = 'AFINN'
GOOGLE_CLOUD_NL = 'GoogleCloud'
REGARD = 'Regard'
PERSPECTIVE = 'Perspective'
EVALUATION_TOOLS = [VADER, AFINN, GOOGLE_CLOUD_NL, REGARD, PERSPECTIVE]

#Perspective
TOXICITY = 'TOXICITY'
SEVERE_TOXICITY = 'SEVERE_TOXICITY'
INSULT = 'INSULT'
IDENTITY_ATTACK = 'IDENTITY_ATTACK'
PROFANITY = 'PROFANITY'
THREAT = 'THREAT'
PERSPECTIVE_CATEGORIES =[TOXICITY, SEVERE_TOXICITY, INSULT, IDENTITY_ATTACK, PROFANITY, THREAT]
#Regard
REGARD_CATEGORIES = ['Negative', 'Neutral', 'Positive']

#Table utils
Y_AXE = ['Binary','Neutral', 'Neo', 'Queer Identity', 'Queer Orientation', 'Queer Other', 'Non-queer Identity', 'Non-queer Orientation', 'Non-queer Other', 'Queer', 'Non-queer']
