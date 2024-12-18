#!/usr/bin/env python
# coding: utf-8

# # Initialisation

# In[1]:


#get_ipython().system('pip3 install torch')


# # Imports

# In[2]:


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


# # Global Variables

# In[3]:


#Data Source
OUTPUT_TEMPLATE = 'output_template/'
TEMPLATES_COMPLETE_PATH = OUTPUT_TEMPLATE + 'template_complete.csv'
DATA_SOURCE = 'dataset_source/'
OUTPUT_PREDICTION = 'output_prediction/'


# In[4]:


url = "http://localhost:11434/api/generate"

class generateSentences():
    def __init__(self, modelName):
            self.modelName = modelName
            self.templateFile = pd.read_csv(TEMPLATES_COMPLETE_PATH, sep=";")
            dicSentences = {
                "Template": [],
                "Generated": []
            }
            print("Generating sentences...")
            for index,row in tqdm(self.templateFile.iterrows(), total=self.templateFile.shape[0], desc='Generating ', unit=' sentences'):
                sentence = row.loc['template']
                #Other propts to try:
                #prompt = f"Complete the follwing sentence and provide the response {sentence}"
                #prompt = "Complete the follwing sentence (return only one response): {0}".format(sentence)
                
                prompt = f"Complete the following sentence: `{sentence}` Provide only the additional words necessary to complete the sentence as output, without repeating the initial part or adding any explanations."
                response = self.modelRequest(prompt)
                dicSentences["Template"].append(sentence)
                dicSentences["Generated"].append(sentence + response)
                print(str(index) +"-"+ sentence + response)
            df = pd.DataFrame.from_dict(dicSentences)    
            #display(df)
            print("Sentences generated!")            
            os.makedirs(OUTPUT_PREDICTION, exist_ok=True)
            df.to_csv(OUTPUT_PREDICTION+self.modelName+'.csv', sep=';', index=False)
            print("File generated!!")
    
    def modelRequest (self, prompt):
        data = {
            "model": self.modelName,
            "prompt": prompt,
            "messages": [
                {
                "role": "user",
                "content": prompt
                }
            ],   
            "stream": False
        }
        
        headers = {
            "Content-Type": 'application/json'
        }

        response = requests.post(url, headers=headers, json=data)
        #return (response.json()['choices'][0]['message']['content'])
        response = response.json()['response'].replace("`", "").replace('\"', '')
        return(response)
    


# In[5]:


Llama3 = 'llama3'
gemma2 = 'gemma2'
generateSentences(Llama3)

