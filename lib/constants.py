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
DATA_SOURCE = 'dataset_source/'
OUTPUT_SENTENCES = 'output_sentences/'
OUTPUT_EVALUATION = 'output_evaluation/'

# TEMPLATE MAP
SUBJECT_ = r'\{marker\} \+ \{subject\}'
TYPE = 'type'
CATEGORY= 'category'
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

#Evaluation Tools
VADER = 'VADER'
AFINN = 'AFINN'
GOOGLE_CLOUD_NL = 'GoogleCloud'
REGARD = 'Regard'
PERSPECTIVE = 'Perspective'

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

# The `clean_response` function removes newline characters, double quotes, and backticks from a given
# response string.
def clean_response(response):
    response = re.sub(r'\n', '', response)
    response = re.sub(r'\"', '', response)
    response = re.sub(r'`', '', response)
    response = response.replace('.', '')
    response = response.replace(r" '", "")
    response = response.replace(r"*", "")
    response = response.replace(r"[", "")
    response = response.replace(r"]", "")
    response = response.lower()
    response = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL)
    #response = re.sub(r'.', '', response)
    #response = f"['{response}']" 
    response = response.split(" ")
    response = response[-1]
    response = response.replace(r"is:", "")
    response = re.sub(r'[^a-zA-Z0-9]', '', response)
    #response = response.replace(r" ", "")
    return response

# The function `getListFromString` takes a string input, removes certain characters, splits the string
# by commas, and returns a list of the resulting elements.
def getListFromString(text):
    text = re.sub(r"'", "", str(text))
    text = re.sub(r'\]', '', text)
    text = re.sub(r'\[', '', text)
    return list(map(str, text.split(",")))

def getCSVFile(folder, modelName, predictionsConsidered):
    files = []
    for f in os.listdir(folder):
        pred = f.replace(f'{modelName}_', '').replace('.csv', '')
        try:
            if re.match(modelName, f) and int(pred) >= predictionsConsidered:
                files.append(int(pred))
        except: 
            continue
    files.sort()
    try:
        return pd.read_csv(f'{folder+modelName}_{files[0]}.csv')
    except Exception as X:
        logger.error("There are no files related to the specified model [{modelName}] with at least {predictionsConsidered} words predicted")

# The function `getListFromString` takes a string containing a list of integers and returns a list of
# integers.
def getListFromString(text):
    text = re.sub(r"'", "", str(text))
    text = re.sub(r'\]', '', text)
    text = re.sub(r'\[', '', text)
    return (map(int, text.split(","))) if "," in text else text

def getTemplateFile(modelName, inputFolder, outputFolder):
    sentenceFile = f"{inputFolder+modelName}.csv"
    evaluationFile = f"{outputFolder+modelName}.csv"
    if os.path.exists(sentenceFile):
        sentenceFile = pd.read_csv(sentenceFile)
        #If the file exists already in the output folder then take that one   
        if os.path.exists(evaluationFile):
            evaluationFile = pd.read_csv(evaluationFile)
            logger.info(f"๏ {evaluationFile.shape[0]} sentences imported!")
            return evaluationFile, sentenceFile[evaluationFile.shape[0]:]
        return pd.DataFrame(), sentenceFile[0:]
    else:
        return None, None