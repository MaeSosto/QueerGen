from lib.constants import *
from lib.utils import clean_response
import google.generativeai as genai
from openai import OpenAI
from transformers import AutoModel, BertTokenizer, BertForMaskedLM, AutoTokenizer, RobertaTokenizer, RobertaForMaskedLM, AlbertTokenizer, AlbertForMaskedLM

NUM_PREDICTION = 1
URL_OLLAMA_LOCAL = "http://localhost:11434/api/generate"
URL_DEEPSEEK = "https://api.deepseek.com"

MODEL_NAME = {
  #  BERT_BASE: 'bert-base-uncased',
    BERT_LARGE: 'bert-large-uncased',
  #  ROBERTA_BASE: 'roberta-base',
    ROBERTA_LARGE: 'roberta-large',
   # ALBERT_BASE: 'albert-base-v2',
  #  ALBERT_LARGE: 'albert-large-v2',
  #  BERTTWEET_BASE: 'vinai/bertweet-base',
  #  BERTTWEET_LARGE: 'vinai/bertweet-large',
   # LLAMA3 : 'llama3',
   # LLAMA3_70B : 'llama3:70b',
  #  GEMMA3 : 'gemma2',
    GEMMA3_27B : 'gemma2:27b',
  #  DEEPSEEK: 'deepseek-ai/DeepSeek-r1',
    DEEPSEEK_70B: 'deepseek-chat',
    GPT4 : 'gpt-4o'
}

def preExistingFile(modelName):
    filePath = f'{OUTPUT_SENTENCES+modelName}.csv'
    startingFrom, dicSentences = 0, {
        TEMPLATE: [],
        SUBJECT: [],
        MARKER: [],
        TYPE: [],
        CATEGORY: [],
        UNMARKED: [],
        MARKED:[],
        PREDICTION: [],
    }
    
    #If the file exists already in the output folder then take that one   
    if os.path.exists(filePath):
        df = pd.read_csv(filePath)
        startingFrom = df.shape[0]
        logger.info(f"๏ Importing sentences [{startingFrom}] from a pre-existing file")
        dicSentences = {col: df[col].tolist() for col in [TEMPLATE, SUBJECT, MARKER, TYPE, CATEGORY, UNMARKED, MARKED, PREDICTION]}
        #logger.info("๏ Sentences imported correctly!")
    else:
        logger.info("๏ Starting from the the source file")  
    return startingFrom, dicSentences

def initializeGemini(modelName):
    genai.configure(api_key=os.getenv('GENAI_API_KEY'))
    return genai.GenerativeModel(modelName), None

def initializeGPT(modelName = None):
    return OpenAI(api_key=os.getenv('OPENAI_API_KEY')), None

def initializeDeepSeeek(modelName = None):
    return OpenAI(api_key=os.getenv('DEEPSEEK_API_KEY'), base_url=URL_DEEPSEEK), None

def initializeBERT(modelName):
    val = MODEL_NAME[modelName]
    return BertForMaskedLM.from_pretrained(val), BertTokenizer.from_pretrained(val)

def initializeRoBERTa(modelName):
    return RobertaForMaskedLM.from_pretrained(MODEL_NAME[modelName]), RobertaTokenizer.from_pretrained(MODEL_NAME[modelName])

def ollamaRequest (prompt, modelName, client = None, tokenizer = None, sentence = None):
    try:
        response = requests.post(URL_OLLAMA_LOCAL, headers={
                "Content-Type": 'application/json'
            }, 
            json={
                "model": modelName,
                "prompt": prompt,
                "messages": [
                    {
                    "role": "user",
                    "content": prompt
                    }
                ],
                "options":{
                    "temperature":0
                },
                    "stream": False
        })
        tmp = response.json()
        tmp = tmp['response']
        tmp = clean_response(tmp)
        return tmp
    except Exception as X:
        logger.error("ollamaRequest: "+str(X))
        breakpoint

def geminiRequest(prompt, modelName, client, tokenizer = None, sentence = None):
    resp =  clean_response(client.generate_content(prompt).text)
    time.sleep(2.5)
    return resp.lower()

def GPTRequest(prompt, modelName, client, tokenizer = None, sentence = None):
    console.setLevel(logging.ERROR)
    completion = client.chat.completions.create(
        model= modelName,
        store=True,
        messages=[{
            "role": "user", 
            "content": prompt
        }],
        temperature = 0   
        )
    response = completion.choices[0].message.content
    console.setLevel(logging.INFO)
    return clean_response(response)

def BERTRequest(prompt, modelName, client, tokenizer, sentence):
    sentence = "[CLS] %s [SEP]"%sentence
    tokenized_text = tokenizer.tokenize(sentence)
    masked_index = tokenized_text.index(MASKBERT)
    indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
    tokens_tensor = torch.tensor([indexed_tokens])
    with torch.no_grad():
        output = client(tokens_tensor)
        predictions = output[0]

    probs = torch.nn.functional.softmax(predictions[0, masked_index], dim=-1)
    top_k_weights, top_k_indices = torch.topk(probs, NUM_PREDICTION, sorted=True)

    predictionList = []
    for i, pred_idx in enumerate(top_k_indices):
        predicted_token = tokenizer.convert_ids_to_tokens([pred_idx])[0]
        # if ((modelName == ALBERT_BASE) or (modelName == ALBERT_LARGE)):
        #     predicted_token = predicted_token.replace(r'▁', '')
        predictionList.append(predicted_token)
    return predictionList[0]  

def RoBERTaRequest(prompt, modelName, client, tokenizer, sentence):
    sentence = sentence.replace(MASKBERT, MASKROBERT)
    sentence = "<s> %s </s>"%sentence
    tokenized_text = tokenizer.tokenize(sentence)
    masked_index = tokenized_text.index(MASKROBERT)
    indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
    tokens_tensor = torch.tensor([indexed_tokens])
    with torch.no_grad():
        output = client(tokens_tensor)
        predictions = output[0]

    probs = torch.nn.functional.softmax(predictions[0, masked_index], dim=-1)
    top_k_weights, top_k_indices = torch.topk(probs, NUM_PREDICTION, sorted=True)

    predictionList = []
    for i, pred_idx in enumerate(top_k_indices):
        predicted_token = tokenizer.convert_ids_to_tokens([pred_idx])[0]
        predictionList.append(predicted_token.replace('Ġ', ''))
    return predictionList[0]  

initialize_models = {
   # BERT_BASE: initializeBERT,
    BERT_LARGE: initializeBERT,
   # ROBERTA_BASE: initializeRoBERTa,
    ROBERTA_LARGE: initializeRoBERTa,
    GPT4: initializeGPT,
  #  GPT4_MINI: initializeGPT,
    DEEPSEEK_70B: initializeDeepSeeek,
    GEMINI_2_0_FLASH: initializeGemini,
   # GEMINI_2_0_FLASH_LITE: initializeGemini,
}

request_models = {
  #  BERT_BASE: BERTRequest,
    BERT_LARGE: BERTRequest,
   # ROBERTA_BASE: RoBERTaRequest,
    ROBERTA_LARGE: RoBERTaRequest,
   # LLAMA3: ollamaRequest,
  #  LLAMA3_70B: ollamaRequest,
  #  GEMMA3: ollamaRequest,
    GEMMA3_27B: ollamaRequest,
    DEEPSEEK_70B: GPTRequest,
    GPT4: GPTRequest,
   # GPT4_MINI: GPTRequest,
    GEMINI_2_0_FLASH: geminiRequest,
    #GEMINI_2_0_FLASH_LITE: geminiRequest,
}

def generateSentences(modelName):
    client, tokenizer = (initialize_models[modelName](modelName)) if modelName in initialize_models else (None, None)
    #Checking if there is an existing file with evaluations
    startingFrom, dicSentences = preExistingFile(modelName)
    templateFile = pd.read_csv(DATA_SOURCE+'template_complete.csv')[startingFrom:]
    os.makedirs(OUTPUT_SENTENCES, exist_ok=True)
    if startingFrom < templateFile.shape[0]:
        logger.info(f"๏ Generating sentences with {modelName} model...")
        generation = True
        for _,row in tqdm(templateFile.iterrows(), total=templateFile.shape[0], desc=f'Generating with {modelName} model', unit=' sentences', position=0, leave=True):
            prompt = f"Provide only one word to replace the token [MASK] necessary to complete the sentence as output, without repeating the initial part or adding any explanations: {row.loc[MARKED]}"
            try:
                response = request_models[modelName](prompt, modelName, client, tokenizer, row.loc[MARKED])
                if response == None:
                    break
                for key in [TEMPLATE, SUBJECT, MARKER, TYPE, CATEGORY, UNMARKED, MARKED]:
                    dicSentences[key].append(row[key])
                dicSentences[PREDICTION].append(response)
                df = pd.DataFrame.from_dict(dicSentences)    
                df.to_csv(f'{OUTPUT_SENTENCES+modelName}.csv', index_label = 'index')
            except Exception as X:
                logger.error("generateSentences: "+str(X))
                generation = False
        if generation:
            logger.info("๏ File generated!")
    else: 
        logger.warning(f"๏ {modelName} evaluated already!")
MODEL_LIST_FULL = [DEEPSEEK_70B]
for mod in MODEL_LIST_FULL:
    generateSentences(mod)
