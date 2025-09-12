# === Imports ===
from src.lib import *
import requests, shutil, subprocess
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
URL_OLLAMA_LOCAL = "http://localhost:11434/api"
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
        self.template_complete_file = pd.read_csv(PATH_DATASET + 'template_complete.csv')
        
        self.func_initialize_model = {
            BERT_BASE: self._initialize_BERT, 
            BERT_LARGE: self._initialize_BERT,
            ROBERTA_BASE: self._initialize_RoBERTa, 
            ROBERTA_LARGE: self._initialize_RoBERTa,
            LLAMA3: self._initialize_Ollama, 
            LLAMA3_70B: self._initialize_Ollama, 
            GEMMA3: self._initialize_Ollama,
            GEMMA3_27B: self._initialize_Ollama, 
            GPT4: self._initialize_GPT, 
            GPT4_MINI: self._initialize_GPT,
            DEEPSEEK: self._initialize_Ollama,
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
        
    def initialize_model(self):
        if self.model_name in self.func_initialize_model: 
            err = self.func_initialize_model[self.model_name]()
            return err
        return False

    def _generate_prediction_from_row(self, row, target_col=None, row_idx=None):
        if self.prompt_num != 3:
            self.sentence = f"{row.loc[MARKED]} {MASKBERT}."
        else: 
            self.sentence = row.loc[MARKED]
        self.prompt = PROMPTS[self.prompt_num].format(self.sentence)

        try:
            response = self.send_request[self.model_name]()
            if response is None:
                return None

            if target_col is not None and row_idx is not None:
                # Used in _check_evaluation_file_integrity
                self.prediction_dic[target_col][row_idx] = response
            else:
                # Used in get_predictions
                for key in [TEMPLATE, SUBJECT, MARKER, TYPE, CATEGORY, UNMARKED, MARKED]:
                    self.prediction_dic[key].append(row[key])
                self.prediction_dic[PREDICTION].append(response)

            self._save_csv(self.prediction_dic)
            return response

        except Exception as e:
            context = "integrity check" if target_col else "prediction"
            logger.error(f"Error during {context}: {e}")
            return None
    
    def _check_generation_file_integrity(self):
        prediction_df = pd.DataFrame.from_dict(self.prediction_dic)

        # for row_idx, row in tqdm(
        #     prediction_df.iterrows(), 
        #     total=self.total_rows,
        #     desc=f"📋 Checking integrity {MODELS_LABELS[self.model_name]} [prompt {self.prompt_num}]"):
        for row_idx, row in prediction_df.iterrows():
            for col in prediction_df.columns:
                value = row[col]

                # if pd.isna(value) or (isinstance(value, str) and value.strip() == ""):
                #     logger.info(f"⚠️ {MODELS_LABELS[self.model_name]} [prompt {int(self.prompt_num)}] missing value at [{row_idx} - {col}]")
                if pd.isna(value):
                    logger.info(f"⚠️ {MODELS_LABELS[self.model_name]} [prompt {int(self.prompt_num)}] NULL value at [{row_idx} - {col}]")
                elif isinstance(value, str) and value.strip().lower() in [""]:
                    logger.info(f"⚠️ {MODELS_LABELS[self.model_name]} [prompt {int(self.prompt_num)}] EMPTY or 'none' string at [{row_idx} - {col}]")
            
                    response = self._generate_prediction_from_row(row, target_col=col, row_idx=row_idx)
                    if response is None:
                        return True  # Abort if request fails
        return False
    
    def get_predictions(self, prompt_num=PROMPT_DEFAULT):
        self.prompt_num = prompt_num
        if prompt_num != 0 and (self.model_name == BERT_BASE or self.model_name == BERT_LARGE or self.model_name == ROBERTA_BASE or self.model_name == ROBERTA_LARGE):
            self.copy_file(f"{PATH_GENERATIONS}prompt_0/{self.model_name}.csv", f"{PATH_GENERATIONS}prompt_{self.prompt_num}/{self.model_name}.csv")
            logger.info(f"✅ {MODELS_LABELS[self.model_name]} [prompt {self.prompt_num}] complete")
            return False
        num_row_processed, self.prediction_dic = self._get_prediction_file()
        self.total_rows = self.template_complete_file.shape[0]

        if num_row_processed >= self.total_rows:
            self._check_generation_file_integrity()
            logger.info(f"✅ {MODELS_LABELS[self.model_name]} [prompt {self.prompt_num}] complete")
            return False

        logger.info(f"🔁 Resuming from row {num_row_processed}")

        for _, row in tqdm(
            self.template_complete_file.iloc[num_row_processed:].iterrows(),
            total=self.total_rows - num_row_processed,
            desc=f"🧬 Generating with {MODELS_LABELS[self.model_name]} [prompt {self.prompt_num}]"
        ):
            response = self._generate_prediction_from_row(row)
            if response is None:
                return True  # Abort if request fails
        
    # === Initialization Functions ===
    def _initialize_BERT(self): 
        val = MODEL_NAME[self.model_name]
        self.client, self.tokenizer = BertForMaskedLM.from_pretrained(val), BertTokenizer.from_pretrained(val)
        return False
    
    def _initialize_RoBERTa(self): 
        self.client, self.tokenizer = RobertaForMaskedLM.from_pretrained(MODEL_NAME[self.model_name]), RobertaTokenizer.from_pretrained(MODEL_NAME[self.model_name])
        return False
    
    def _initialize_Gemini(self): 
        api_key = os.getenv('GENAI_API_KEY')
        if api_key is None:
            logger.error(f"⚠️ GENAI_API_KEY is missing")
            return True
        genai.configure(api_key=api_key) 
        self.client, self.tokenizer = genai.GenerativeModel(self.model_name), None
        return False
    
    def _initialize_GPT(self): 
        api_key = os.getenv('OPENAI_API_KEY')
        if api_key is None:
            logger.error(f"⚠️ OPENAI_API_KEY is missing")
            return True
        self.client, self.tokenizer = OpenAI(api_key=api_key), None
        return False
    
    def _initialize_Ollama(self):
        logger.info(f"🚦 Model '{self.model_name}' is testing")
        if not self.check_ollama_server():
            if not self.check_model_is_downloaded():
                return False
        return True
    
    def _initialize_DeepSeeek(self): 
        api_key = os.getenv('DEEPSEEK_API_KEY')
        if api_key is None:
            logger.error(f"⚠️ DEEPSEEK_API_KEY is missing")
            return True
        self.client, self.tokenizer = OpenAI(api_key=api_key, base_url=URL_DEEPSEEK), None
        return False
    
    def _get_prediction_file(self):
        prediction_file_path = f'{PATH_GENERATIONS}prompt_{self.prompt_num}/{self.model_name}.csv'
        if os.path.exists(prediction_file_path): #If file exist read file
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
            response = requests.post(f"{URL_OLLAMA_LOCAL}/generate", headers={"Content-Type": 'application/json'}, json={
                "model": self.model_name,
                "prompt": self.prompt,
                "messages": [{"role": "user", "content": self.prompt}],
                "options": {"temperature": 0},
                "stream": False
            })
            response = response.json()['response']
            if response == None or response == "":
                logger.error(f"_request_ollama: {response}")
                return None
            return self._clean_response(response)
        
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

    def check_ollama_server(self):
        try:
            response = requests.get(f"{URL_OLLAMA_LOCAL}/tags")
            if not(response.status_code == 200):
                logger.error(f"⚠️ Ollama server is not running")
                return True
            return False
        except requests.RequestException:
            logger.error(f"⚠️ Ollama server is not running")
            return True
        
    def check_model_is_downloaded(self):
        try:
            # Step 1: Check if model is downloaded
            list_output = subprocess.check_output(["ollama", "list"], text=True)
            list_output = self.clean_ollama_list(list_output)
            if self.model_name not in list_output:
                print(f"❌ Model '{MODELS_LABELS(self.model_name)}' not found in Ollama. Try: ollama pull {self.model_name}")
                return True
            
            # # Step 2: Test if model is functional with a small prompt
            # test_prompt = "Hello, are you working?"
            # response = requests.post(f"{URL_OLLAMA_LOCAL}/generate", headers={"Content-Type": 'application/json'}, json={
            #     "model": self.model_name,
            #     "prompt": test_prompt,
            #     "messages": [{"role": "user", "content": test_prompt}],
            #     "options": {"temperature": 0},
            #     "stream": False
            # })
            # if not(response.status_code == 200):
            #     logger.error(f"⚠️ {MODEL_NAME[self.model_name]} server is not running")
            #     return True
            return False
            # subprocess.run(
            #     ["ollama", "run", self.model_name],
            #     input=test_prompt,
            #     text=True,
            #     check=True,
            #     capture_output=True
            # )
        
        except subprocess.CalledProcessError as e:
            self.model_name(f"⚠️ Error running model '{self.model_name}': {e}")
            return True
        except FileNotFoundError:
            logging.error("❌ Ollama is not installed or not in PATH.")
            return True
    
    def clean_ollama_list(self, list):
        #list = list.replace("NAME                  ID              SIZE      MODIFIED    ", "")
        list = list.split("\n")
        del list[0]
        list = [elem.split(" ", 1)[0].replace(":latest", "") for elem in list]
        list = [s for s in list if s != ""]
        return list
    
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
        return str(response)
    
    def _save_csv(self, prediction_dic):
        os.makedirs(f"{PATH_GENERATIONS}prompt_{self.prompt_num}/", exist_ok=True)
        pd.DataFrame.from_dict(prediction_dic).to_csv(f"{PATH_GENERATIONS}prompt_{self.prompt_num}/{self.model_name}.csv", index_label='index')
        

    def copy_file(self, input_path, output_path):
        # Ensure the source file exists
        if not os.path.isfile(input_path):
            logger.error(f"Source file not found: {input_path}")
            return True
        # Ensure the output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Copy the file
        shutil.copy2(input_path, output_path)  # copy2 preserves metadata
        #print(f"Copied {input_path} -> {output_path}")
        return False
    
        
    