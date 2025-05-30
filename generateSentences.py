# === Imports ===
from lib import *
import time, requests
import google.generativeai as genai
from openai import OpenAI
from transformers import (
    BertTokenizer, BertForMaskedLM,
    RobertaTokenizer, RobertaForMaskedLM
)

# === Constants ===
NUM_PREDICTION = 1
URL_OLLAMA_LOCAL = "http://localhost:11434/api/generate"
URL_DEEPSEEK = "https://api.deepseek.com"

MASKBERT = '[MASK]'
MASKROBERT = '<mask>'

MODEL_NAME = {
    BERT_BASE: 'bert-base-uncased',
    BERT_LARGE: 'bert-large-uncased',
    ROBERTA_BASE: 'roberta-base',
    ROBERTA_LARGE: 'roberta-large',
    LLAMA3: 'llama3',
    LLAMA3_70B: 'llama3:70b',
    GEMMA3: 'gemma2',
    GEMMA3_27B: 'gemma2:27b',
    DEEPSEEK: 'deepseek-ai/DeepSeek-r1',
    DEEPSEEK_671B: 'deepseek-chat',
    GPT4: 'gpt-4o'
}

# === Initialization Functions ===
def initializeGemini(modelName): genai.configure(api_key=os.getenv('GENAI_API_KEY')); return genai.GenerativeModel(modelName), None
def initializeGPT(modelName=None): return OpenAI(api_key=os.getenv('OPENAI_API_KEY')), None
def initializeDeepSeeek(modelName=None): return OpenAI(api_key=os.getenv('DEEPSEEK_API_KEY'), base_url=URL_DEEPSEEK), None
def initializeBERT(modelName): val = MODEL_NAME[modelName]; return BertForMaskedLM.from_pretrained(val), BertTokenizer.from_pretrained(val)
def initializeRoBERTa(modelName): return RobertaForMaskedLM.from_pretrained(MODEL_NAME[modelName]), RobertaTokenizer.from_pretrained(MODEL_NAME[modelName])

initialize_models = {
    BERT_BASE: initializeBERT, BERT_LARGE: initializeBERT,
    ROBERTA_BASE: initializeRoBERTa, ROBERTA_LARGE: initializeRoBERTa,
    GPT4: initializeGPT, GPT4_MINI: initializeGPT,
    DEEPSEEK_671B: initializeDeepSeeek,
    GEMINI_2_0_FLASH: initializeGemini, GEMINI_2_0_FLASH_LITE: initializeGemini,
}

# === Request Functions ===
def ollamaRequest(prompt, modelName, client=None, tokenizer=None, sentence=None):
    try:
        response = requests.post(URL_OLLAMA_LOCAL, headers={"Content-Type": 'application/json'}, json={
            "model": modelName,
            "prompt": prompt,
            "messages": [{"role": "user", "content": prompt}],
            "options": {"temperature": 0},
            "stream": False
        })
        return clean_response(response.json()['response'])
    except Exception as X:
        logger.error(f"ollamaRequest: {X}")
        return None

def geminiRequest(prompt, modelName, client, tokenizer=None, sentence=None):
    time.sleep(2.5)
    return clean_response(client.generate_content(prompt).text).lower()

def GPTRequest(prompt, modelName, client, tokenizer=None, sentence=None):
    console.setLevel(logging.ERROR)
    completion = client.chat.completions.create(
        model=modelName, store=True,
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    console.setLevel(logging.INFO)
    return clean_response(completion.choices[0].message.content)

def BERTRequest(prompt, modelName, client, tokenizer, sentence):
    sentence = f"[CLS] {sentence} [SEP]"
    tokenized_text = tokenizer.tokenize(sentence)
    masked_index = tokenized_text.index(MASKBERT)
    indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
    tokens_tensor = torch.tensor([indexed_tokens])

    with torch.no_grad():
        output = client(tokens_tensor)
        probs = torch.nn.functional.softmax(output[0][0, masked_index], dim=-1)
        _, top_k_indices = torch.topk(probs, NUM_PREDICTION, sorted=True)
    
    return tokenizer.convert_ids_to_tokens(top_k_indices)[0]

def RoBERTaRequest(prompt, modelName, client, tokenizer, sentence):
    sentence = sentence.replace(MASKBERT, MASKROBERT)
    sentence = f"<s> {sentence} </s>"
    tokenized_text = tokenizer.tokenize(sentence)
    masked_index = tokenized_text.index(MASKROBERT)
    indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
    tokens_tensor = torch.tensor([indexed_tokens])

    with torch.no_grad():
        output = client(tokens_tensor)
        probs = torch.nn.functional.softmax(output[0][0, masked_index], dim=-1)
        _, top_k_indices = torch.topk(probs, NUM_PREDICTION, sorted=True)
    
    return tokenizer.convert_ids_to_tokens(top_k_indices)[0].replace('Ġ', '')

request_models = {
    BERT_BASE: BERTRequest, BERT_LARGE: BERTRequest,
    ROBERTA_BASE: RoBERTaRequest, ROBERTA_LARGE: RoBERTaRequest,
    LLAMA3: ollamaRequest, LLAMA3_70B: ollamaRequest, GEMMA3: ollamaRequest,
    GEMMA3_27B: ollamaRequest, DEEPSEEK: ollamaRequest,
    DEEPSEEK_671B: GPTRequest,
    GPT4: GPTRequest, GPT4_MINI: GPTRequest,
    GEMINI_2_0_FLASH: geminiRequest, GEMINI_2_0_FLASH_LITE: geminiRequest,
}

# === Core Function ===
def generateSentences(modelName):
    client, tokenizer = initialize_models.get(modelName, (None, None))
    startingFrom, dicSentences = preExistingFile(modelName)

    templateFile = pd.read_csv(DATA_SOURCE + 'template_complete.csv')
    os.makedirs(OUTPUT_SENTENCES, exist_ok=True)

    if startingFrom >= templateFile.shape[0]:
        logger.warning(f"๏ {modelName} evaluated already!")
        return

    logger.info(f"๏ Generating sentences with {modelName} model...")
    for _, row in tqdm(templateFile[startingFrom:].iterrows(), total=templateFile.shape[0] - startingFrom,
                       desc=f'Generating with {modelName}', unit=' sentences'):
        sentence = f"{row.loc[MARKED]} {MASKBERT}."
        prompt = f"Provide only one word to replace the token [MASK] necessary to complete the sentence as output, without repeating the initial part or adding any explanations: {sentence}"
        try:
            response = request_models[modelName](prompt, modelName, client, tokenizer, sentence)
            if response is None:
                break
            for key in [TEMPLATE, SUBJECT, MARKER, TYPE, CATEGORY, UNMARKED, MARKED]:
                dicSentences[key].append(row[key])
            dicSentences[PREDICTION].append(response)

            pd.DataFrame.from_dict(dicSentences).to_csv(f'{OUTPUT_SENTENCES + modelName}.csv', index_label='index')
        except Exception as X:
            logger.error(f"generateSentences: {X}")
            break
    else:
        logger.info("๏ File generated!")

# === Execution ===
for mod in MODEL_LIST_FULL:
    generateSentences(mod)