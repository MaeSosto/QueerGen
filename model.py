# === Imports ===
from lib import *
import time, requests
import google.generativeai as genai
from openai import OpenAI
from transformers import (
    BertTokenizer, BertForMaskedLM,
    RobertaTokenizer, RobertaForMaskedLM
)
from transformers import logging
logging.set_verbosity_error()

# === Constants ===
NUM_PREDICTION = 1
URL_OLLAMA_LOCAL = "http://localhost:11434/api/generate"
URL_DEEPSEEK = "https://api.deepseek.com"

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
        
class Model:
    
    def __init__(self, model_name):
        self.model_name = model_name
        self.template_complete_file = pd.read_csv(DATA_SOURCE + 'template_complete.csv')
        
        self.initialize_model = {
            BERT_BASE: self._initialize_BERT, 
            BERT_LARGE: self._initialize_BERT,
            ROBERTA_BASE: self._initialize_RoBERTa, 
            ROBERTA_LARGE: self._initialize_RoBERTa,
            GPT4: self._initialize_GPT, 
            GPT4_MINI: self._initialize_GPT,
            DEEPSEEK_671B: self._initialize_DeepSeeek,
            GEMINI_2_0_FLASH: self._initialize_Gemini, 
            GEMINI_2_0_FLASH_LITE: self._initialize_Gemini,
        }
        
        self.send_request = {
            BERT_BASE: self._request_BERT, 
            BERT_LARGE: self._request_BERT,
            ROBERTA_BASE: self._request_RoBERTa, 
            ROBERTA_LARGE: self._request_RoBERTa,
            LLAMA3: self._request_ollama, 
            LLAMA3_70B: self._request_ollama, 
            GEMMA3: self._request_ollama,
            GEMMA3_27B: self._request_ollama, 
            DEEPSEEK: self._request_ollama,
            DEEPSEEK_671B: self._request_open_ai,
            GPT4: self._request_open_ai, 
            GPT4_MINI: self._request_open_ai,
            GEMINI_2_0_FLASH: self._request_gemini, 
            GEMINI_2_0_FLASH_LITE: self._request_gemini,
        }
        
        if self.model_name in self.initialize_model: 
            self.initialize_model[self.model_name]()
    
    def get_predictions(self, prompt_num = 2):
        self.prompt_num = prompt_num
        num_row_processed, prediction_dic = self._get_prediction_file()
        
        if num_row_processed >= self.template_complete_file.shape[0]:
            logger.info(f"✅ {self.model_name} [prompt {self.prompt_num}]")
            return num_row_processed
        else:
            logger.info(f"🔁 Importing sentences [{num_row_processed}]")
            

            #logger.info(f"๏ Generating sentences with {self.prompt_num} and {self.model_name} model...")
            for _, row in tqdm(self.template_complete_file[num_row_processed:].iterrows(), total= self.template_complete_file.shape[0] - num_row_processed, desc=f"🧬 Generating with {self.model_name} [prompt {self.prompt_num}]"):
                self.sentence = f"{row.loc[MARKED]} {MASKBERT}."
                self.prompt = PROMPTS[prompt_num].format(self.sentence)
                try:
                    response = self.send_request[self.model_name]()
                    if response is None:
                        return None
                    for key in [TEMPLATE, SUBJECT, MARKER, TYPE, CATEGORY, UNMARKED, MARKED]:
                        prediction_dic[key].append(row[key])
                    prediction_dic[PREDICTION].append(response)

                    self._save_csv(prediction_dic)
                except Exception as X:
                    logger.error(f"generateSentences: {X}")
                    break
            #logger.info("๏ File generated!")

        
    # === Initialization Functions ===
    def _initialize_BERT(self): 
        val = MODEL_NAME[self.model_name]
        self.client, self.tokenizer = BertForMaskedLM.from_pretrained(val), BertTokenizer.from_pretrained(val)
    
    def _initialize_RoBERTa(self): 
        self.client, self.tokenizer = RobertaForMaskedLM.from_pretrained(MODEL_NAME[self.model_name]), RobertaTokenizer.from_pretrained(MODEL_NAME[self.model_name])
    
    def _initialize_Gemini(self): 
        genai.configure(api_key=os.getenv('GENAI_API_KEY')) 
        self.client, self.tokenizer = genai.GenerativeModel(self.model_name), None
    
    def _initialize_GPT(self): 
        api_key = os.getenv('OPENAI_API_KEY')
        self.client, self.tokenizer = OpenAI(api_key=api_key), None
    
    def _initialize_DeepSeeek(self): 
        self.client, self.tokenizer = OpenAI(api_key=os.getenv('DEEPSEEK_API_KEY'), base_url=URL_DEEPSEEK), None
    
    def _get_prediction_file(self):
        prediction_file_path = f'{OUTPUT_SENTENCES}prompt_{self.prompt_num}/{self.model_name}.csv'
        if os.path.exists(prediction_file_path):
            prediction_file = pd.read_csv(prediction_file_path)
            num_row_processed = prediction_file.shape[0]
            prediction_dic = {col: prediction_file[col].tolist() for col in [TEMPLATE, SUBJECT, MARKER, TYPE, CATEGORY, UNMARKED, MARKED, PREDICTION]}
        else:
            logger.info("○ Starting from 0")
            num_row_processed = 0
            prediction_dic = {key: [] for key in [TEMPLATE, SUBJECT, MARKER, TYPE, CATEGORY, UNMARKED, MARKED, PREDICTION]}   
        return num_row_processed, prediction_dic
    
    # === Request Functions ===
    def _request_BERT(self):
        try:
            sentence = f"[CLS] {self.sentence} [SEP]"
            tokenized_text = self.tokenizer.tokenize(sentence)
            masked_index = tokenized_text.index(MASKBERT)
            indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokenized_text)
            tokens_tensor = torch.tensor([indexed_tokens])

            with torch.no_grad():
                output = self.client(tokens_tensor)
                probs = torch.nn.functional.softmax(output[0][0, masked_index], dim=-1)
                _, top_k_indices = torch.topk(probs, NUM_PREDICTION, sorted=True)
            
            return self.tokenizer.convert_ids_to_tokens(top_k_indices)[0]
        except Exception as X:
            logger.error(f"_request_BERT: {X}")
            return None

    def _request_RoBERTa(self):
        try:
            sentence = self.sentence.replace(MASKBERT, MASKROBERT)
            sentence = f"<s> {sentence} </s>"
            tokenized_text = self.tokenizer.tokenize(sentence)
            masked_index = tokenized_text.index(MASKROBERT)
            indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokenized_text)
            tokens_tensor = torch.tensor([indexed_tokens])

            with torch.no_grad():
                output = self.client(tokens_tensor)
                probs = torch.nn.functional.softmax(output[0][0, masked_index], dim=-1)
                _, top_k_indices = torch.topk(probs, NUM_PREDICTION, sorted=True)
            
            return self.tokenizer.convert_ids_to_tokens(top_k_indices)[0].replace('Ġ', '')
        except Exception as X:
            logger.error(f"_request_RoBERTa: {X}")
            return None
    
    def _request_ollama(self):
        try:
            response = requests.post(URL_OLLAMA_LOCAL, headers={"Content-Type": 'application/json'}, json={
                "model": self.model_name,
                "prompt": self.prompt,
                "messages": [{"role": "user", "content": self.prompt}],
                "options": {"temperature": 0},
                "stream": False
            })
            return self._clean_response(response.json()['response'])
        
        except Exception as X:
            logger.error(f"_request_ollama: {response['text']}")
            return None

    def _request_gemini(self):
        #time.sleep(2.5)
        try:
            return self._clean_response(self.client.generate_content(self.prompt).text).lower()
        except Exception as X:
            logger.error(f"_request_gemini: {X}")
            return None

    def _request_open_ai(self):
        logger.setLevel(logging.ERROR)
        try:
            completion = self.client.chat.completions.create(
                model=self.model_name, store=True,
                messages=[{"role": "user", "content": self.prompt}],
                temperature=0
            )
            logger.setLevel(logging.INFO)
            return self._clean_response(completion.choices[0].message.content)
        except Exception as X:
            logger.error(f"_request_open_ai: {X}")
            return None
        
    # === Utils ===
    def _clean_response(self, response):
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
        response = response.split(" ")
        response = response[-1]
        response = response.replace(r"is:", "")
        response = re.sub(r'[^a-zA-Z0-9]', '', response)
        return response
    
    def _save_csv(self, prediction_dic):
        os.makedirs(f"{OUTPUT_SENTENCES}prompt_{self.prompt_num}/", exist_ok=True)
        pd.DataFrame.from_dict(prediction_dic).to_csv(f"{OUTPUT_SENTENCES}prompt_{self.prompt_num}/{self.model_name}.csv", index_label='index')
    
        
    