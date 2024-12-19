import torch
import logging
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
THE = 'the'

# # Build Complete Template
class CompleteTemplateBuilder():
    def __init__(self):
        self.template = pd.read_csv(TEMPLATE_PATH, sep=";")
        self.nouns = pd.read_csv(NOUNS_PATH, sep=';')
        self.template_builder()

    def plural_form(self, be, sentence):
        if be == 'are':
            word = sentence.split(" ")[1]
            if word[-1] == 's':
                sentence = sentence.replace(word, word[:-1])
        return sentence

    def template_builder(self):
        dataList =[]
        for index, row in tqdm(self.template.iterrows(), total=self.template.shape[0], desc='Creating template', unit=' sentences'):
            sentence = row.loc['template']
            
            #Creating sentences with nouns
            for ind, r in self.nouns.iterrows():
                _sentence = sentence.replace(TARGET_, f"{'The' if sentence.split(TARGET_)[0] == '' else 'the'} {r.loc[SUBJECT]} person") if r.loc[THE] == 'y' else sentence.replace(TARGET_, f"{'The' if sentence.split(TARGET_)[0] == '' else 'the'} {r.loc[SUBJECT]}")
                _sentence = _sentence.replace(BE_, 'is').replace(WERE_, 'was').replace(HAVE_, 'has')
                
                dataList.append([
                    index,
                    _sentence, #new template
                    r.loc[TYPE], #type
                    r.loc[CATEGORY], #category
                    r.loc[SUBJECT] #subject
                ]) 

        data_df = pd.DataFrame(dataList, columns=["index","template", TYPE, CATEGORY, SUBJECT])
        #display(data_df)
        print(data_df)
        os.makedirs(OUTPUT_TEMPLATE, exist_ok=True)
        data_df.to_csv(TEMPLATES_COMPLETE_PATH, sep=';', index=False)

CompleteTemplateBuilder()

