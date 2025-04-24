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
logging.basicConfig(filename='logFile.log', encoding='utf-8', level=logging.DEBUG)
console = logging.StreamHandler()
console.setLevel(logging.INFO)
logger.addHandler(console)
#print(f"PyTorch version: {torch.__version__}")

# Set the device      
device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cuda") if torch.cuda.is_available() else torch.device('cpu')
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
#BERT_BASE = 'BERT_base'
BERT_LARGE = 'BERT_large'
#ROBERTA_BASE = 'RoBERTa_base'
ROBERTA_LARGE = 'RoBERTa_large'
#ALBERT_BASE = 'AlBERT_base'
#ALBERT_LARGE = 'AlBERT_large'
#BERTTWEET_BASE = 'BERTweet_base'
#BERTTWEET_LARGE = 'BERTweet_large'
#LLAMA3 = 'llama3'
#LLAMA3_70B = 'llama3:70b'
LLAMA3_3 = 'llama3.3'
#GEMMA3 = 'gemma3'
GEMMA3_27B = 'gemma3:27b'
#GEMINI_1_5_FLASH = "gemini-1.5-flash"
GEMINI_2_0_FLASH = "gemini-2.0-flash"
#GEMINI_2_0_FLASH_LITE = "gemini-2.0-flash-lite"
#GPT4_MINI = 'gpt-4o-mini'
GPT4 = 'gpt-4o'
#DEEPSEEK = 'deepseek-r1'
DEEPSEEK_70B = 'deepseek-reasoner'
#LIGHT_MODELS = [BERT_BASE, BERT_LARGE, LLAMA3, GEMMA3, GPT4, GPT4_MINI, GEMINI_2_0_FLASH, GEMINI_2_0_FLASH_LITE]
#HEAVY_MODELS = [LLAMA3_70B, GEMMA3_27B, DEEPSEEK, DEEPSEEK_70B]
#MODEL_LIST_FULL = [BERT_BASE, BERT_LARGE, LLAMA3, LLAMA3_70B, GEMMA3, GEMMA3_27B, DEEPSEEK, GPT4, GPT4_MINI, GEMINI_2_0_FLASH, GEMINI_2_0_FLASH_LITE]

MODELS_LABELS = {
 #   BERT_BASE : 'BERT Base',
    BERT_LARGE : 'BERT Large',
 #   ROBERTA_BASE : 'RoBERTa Base',
    ROBERTA_LARGE : 'RoBERTa Large',
  #  ALBERT_BASE : 'AlBERT Base',
   # ALBERT_LARGE : 'AlBERT Large',
   # BERTTWEET_BASE : 'BERTweet Base',
   # BERTTWEET_LARGE : 'BERTweet Large',
   # LLAMA3 : 'Llama 3',
   # LLAMA3_70B : 'Llama 3(70b)',
   # GEMMA3 : 'Gemma 3',
    GEMMA3_27B : 'Gemma 3(27b)',
    #GEMINI_1_5_FLASH : "Gemini 1.5 Flash",
    GEMINI_2_0_FLASH : "Gemini 2.0 Flash",
    #GEMINI_2_0_FLASH_LITE : "Gemini 2.0 Flash Lite",
    #GPT4_MINI : 'GPT4o Mini',
    GPT4 : 'GPT4o',
    DEEPSEEK_70B: 'DeepSeek R1'
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
SUBJ_CATEGORIES = [UNMARKED, QUEER, NONQUEER]
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
PERSPECTIVE_MAP = {
    TOXICITY: "Toxicity", 
    SEVERE_TOXICITY: "Severe Toxicity", 
    INSULT: "Insult", 
    IDENTITY_ATTACK: "Identity Attack", 
    PROFANITY: "Profanity", 
    THREAT: "Threat"
}
#Regard
REGARD_CATEGORIES = ['Positive', 'Neutral', 'Negative']

#Table utils
Y_AXE = ['Binary','Neutral', 'Neo', 'Queer Identity', 'Queer Orientation', 'Queer Other', 'Non-queer Identity', 'Non-queer Orientation', 'Non-queer Other', 'Queer', 'Non-queer']
