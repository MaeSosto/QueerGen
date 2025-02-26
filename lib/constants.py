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

logger = logging.getLogger()
logging.basicConfig(level=logging.ERROR)# OPTIONAL
logger.setLevel(logging.ERROR)
#print(f"PyTorch version: {torch.__version__}")

# Set the device      
device = "mps" if torch.backends.mps.is_available() else torch.device("cuda") if torch.cuda.is_available() else torch.device('cpu')
#print(f"Using device: {device}")

# # Global Variables
#Data Source
DATA_SOURCE = 'dataset_source/'
OUTPUT_TEMPLATE = 'output_template/'
OUTPUT_QUEERBENCH = 'output_queerbench/'
OUTPUT_GRAPHS = 'output_graphs/'
OUTPUT_PREDICTION = 'output_prediction/'
OUTPUT_SENTENCES = 'output_sentences/'
OUTPUT_EVALUATION = 'output_evaluation/'
EVALUATION_MEASUREMENT_PATH = '.venv/evaluate-main/measurements/'
EVALUATION_METRICS_PATH = '.venv/evaluate-main/metrics/'

# TEMPLATE MAP
BE_ = '<be>'
TARGET_ = '<target>'
HAVE_ = '<have>'
WERE_ = '<were>'
QUEERNESS = 'queerness'
TYPE = 'type'
CATEGORY= 'category'
SUBJECT = 'subject'
MASKBERT_ = r'\[MASK\]'
MASKBERT = '[MASK]'
MASKROBERT = '<mask>'
THE = 'the'
PERSON = 'person'
TEMPLATE = 'template'
GENERATED = 'generated'
ORIGINAL = 'original'

# MODELS
BERT_BASE = 'BERT_base'
BERT_LARGE = 'BERT_large'
ROBERTA_BASE = 'RoBERTa_base'
ROBERTA_LARGE = 'RoBERTa_large'
ALBERT_BASE = 'AlBERT_base'
ALBERT_LARGE = 'AlBERT_large'
BERTTWEET_BASE = 'BERTweet_base'
BERTTWEET_LARGE = 'BERTweet_large'
LLAMA3 = 'llama3'
LLAMA3_70B = 'llama3:70b'
LLAMA3_3 = 'llama3.3'
GEMMA2 = 'gemma2'
GEMMA2_27B = 'gemma2:27b'
GEMINI_FLASH = "gemini-1.5-flash"
GPT4_MINI = 'gpt-4o-mini'
GPT4 = 'gpt-4o'
MODEL_LIST = [BERT_BASE, BERT_LARGE, ROBERTA_BASE, ROBERTA_LARGE, ALBERT_BASE, ALBERT_LARGE, BERTTWEET_BASE, BERTTWEET_LARGE, LLAMA3, LLAMA3_70B, GEMMA2, GEMMA2_27B, GPT4, GPT4_MINI]

# TEMPLATE MAP
NOUN = 'noun'
PRONOUN = 'pronoun'
CATEGORY = 'category'
NAME = 'name'
TYPE = 'type'
BINARY = 'binary'
NEO = 'neo'
NEUTRAL = 'neutral'
QUEER = 'queer'
NONQUEER = 'non-queer'
ORIENTATION = 'orientation'
IDENTITY = 'identity'
OTHER = 'other'
AVERAGE = 'Avg'
STDEV = 'St. Dev'
TOTAL = 'Tot'
QUEERBENCH = 'QueerBench'
SUBJECT_TYPE = [NOUN, PRONOUN]
PRONOUN_CATEGORIES = [BINARY, NEO, NEUTRAL]
NOUN_TYPES = [QUEER, NONQUEER, NEUTRAL]
NOUN_CATEGORIES = [ORIENTATION, IDENTITY, OTHER]

#Evaluation Tools
VADER = 'VADER'
TEXTBLOB = 'textBlob'
AFINN = 'AFINN'
HURTLEX = 'HurtLex'
GOOGLE_CLOUD_NL = 'GoogleCloud'
PERSPECTIVE = 'Perspective'
PERPLEXITY = 'Perplexity'
PERPLEXITY_PERS = 'Perplexity (PERS)'
SURPRISAL = 'Surprisal'
SURPRISAL_PERS = 'Surprisal (PERS)'
HONEST = 'HONEST'
TOXICITY = 'toxicity'
REGARD = 'Regard'
EVALUATION_TOOLS = [VADER, TEXTBLOB, AFINN, GOOGLE_CLOUD_NL, REGARD, TOXICITY, PERSPECTIVE, HURTLEX, HONEST, PERPLEXITY, PERPLEXITY_PERS, SURPRISAL, SURPRISAL_PERS]

# HurtLex
HURTLEX_CATEGORIES = ['Animals', 'Crime and immoral behavior', 'Derogatory words', 'Female genitalia', 'Male genitalia', 'Homosexuality', 'Potential negative connotations', 'Professions and occupations', 'Prostitution', 'Social and economic disadvantage']
HURTLEX_CATEGORIES_SHORT = ['an', 're', 'cds', 'asf', 'asm', 'om', 'qas', 'pa', 'pr', 'is']

#Perspective
TOXICITY = 'TOXICITY'
SEVERE_TOXICITY = 'SEVERE_TOXICITY'
INSULT = 'INSULT'
IDENTITY_ATTACK = 'IDENTITY_ATTACK'
PROFANITY = 'PROFANITY'
THREAT = 'THREAT'
PERSPECTIVE_CATEGORIES =[TOXICITY, SEVERE_TOXICITY, INSULT, IDENTITY_ATTACK, PROFANITY, THREAT]

#Regard
POSITIVE = 'positive'
NEUTRAL = 'neutral'
NEGATIVE = 'negative'
REGARD_CATEGORIES = [POSITIVE, NEGATIVE, NEUTRAL, OTHER]

#Table utils
Y_AXE = ['Binary','Neutral', 'Neo', 'Queer Identity', 'Queer Orientation', 'Queer Other', 'Non-queer Identity', 'Non-queer Orientation', 'Non-queer Other', 'Queer', 'Non-queer']
