# === Imports ===
from lib import *
from afinn import Afinn
from evaluate import load 
import warnings, time
from googleapiclient.discovery import build
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from flair.nn import Classifier
from flair.data import Sentence
from textblob import TextBlob
from googleapiclient.errors import HttpError
from transformers import BertTokenizer, BertForMaskedLM
from transformers import logging
logging.set_verbosity_error()
warnings.filterwarnings('ignore')

# === Constants ===
EVALUATION_MEASUREMENT_PATH = '.venv/evaluate/measurements/'

class Evaluation:
    
    def __init__(self):
        self.initialize_tools = {
            # AFINN: self._afinn_setup,
            VADER: self._VADER_setup,
            # FLAIR: self._FLAIR_setup,
            REGARD: self._regard_setup,
            PERSPECTIVE: self._perpective_setup,
            LOG_LIKELIHOOD: self._log_likelyhood_setup
        }
        
        self.tool_functions = {
            # AFINN: self._get_Afinn_scores,
            VADER: self._get_VADER_scores,
            # FLAIR: self._get_FLAIR_scores,
            # TEXTBLOB: self._get_TextBlob_scores,
            REGARD: self._get_regard_scores,
            PERSPECTIVE: self._get_perspective_scores,
            LOG_LIKELIHOOD: self._get_log_likelyhood_scores,
        }
        
    def evaluate(self, model_name, prompt_num = "prompt_0"):
        self.model_name = model_name
        self.prompt_num = prompt_num
        os.makedirs(OUTPUT_EVALUATION + self.prompt_num+"/", exist_ok=True)
        
        self.evaluation_file = self._get_template_file()
        if self.evaluation_file.empty:
            return None
        
        self.predictions_list = [str(row[PREDICTION]) for _, row in self.evaluation_file.iterrows()]
        self.unmarked_sentence_list = [f"{row[UNMARKED]} {row[PREDICTION]}" for _, row in self.evaluation_file.iterrows()]
        self.xyz_subject = [f"{re.sub('The '+SUBJECT_, 'xyz', row[TEMPLATE])} {row[PREDICTION]}." for _, row in self.evaluation_file.iterrows()]
        
        start_evaluation = True
        for key, score_function in self.tool_functions.items():
            if start_evaluation: logger.info(f"○ Evaluating {model_name} with {prompt_num}"); start_evaluation = False
            if key in self.initialize_tools:
                self.initialize_tools[key]()
            self.key = key
            if key == REGARD and not any(f"{key} {cat}" in self.evaluation_file.columns for cat in REGARD_CATEGORIES):
                logger.info(f"○ Calculating {key} scores...")
                res = score_function()
                if res: break
            elif key == PERSPECTIVE and not any(f"{key} {cat}" in self.evaluation_file.columns for cat in PERSPECTIVE_CATEGORIES):
                logger.info(f"○ Calculating {key} scores...")
                res = score_function()
                if res: break
            elif key != REGARD and key != PERSPECTIVE and key not in self.evaluation_file.columns:
                logger.info(f"○ Calculating {key} scores...")
                score_function()
                
            self.save_csv()
        logger.info(f"○ Evaluated: {model_name} with {self.prompt_num}")    
        
    def _get_template_file(self):
        prediction_file = f"{OUTPUT_SENTENCES}{self.prompt_num}/{self.model_name}.csv"
        evaluation_file = f"{OUTPUT_EVALUATION}{self.prompt_num}/{self.model_name}.csv"

        if os.path.exists(prediction_file):
            prediction_file = pd.read_csv(prediction_file)
            if os.path.exists(evaluation_file):
                evaluation_file = pd.read_csv(evaluation_file)
                logger.info(f"๏ {evaluation_file.shape[0]} sentences imported!")
                prediction_file = prediction_file[evaluation_file.shape[0]:]
            else:
                evaluation_file, prediction_file = pd.DataFrame(), prediction_file[0:]
            prediction_file = pd.concat([evaluation_file, prediction_file]) if not evaluation_file.empty else prediction_file
            return prediction_file
        else: 
            logger.warning(f"○ {self.model_name} with {self.prompt_num} prediction file not found")
            return pd.DataFrame()
    
    def save_csv(self):
        if self.model_name in MODEL_MLM:
            for prompt in PROMPTS:
                self.evaluation_file.to_csv(
                    OUTPUT_EVALUATION + prompt + "/" + self.model_name + ".csv",
                    index=False
                )
        else:
            self.evaluation_file.to_csv(
                OUTPUT_EVALUATION + self.prompt_num + "/" + self.model_name + ".csv",
                index=False
            )
            
    # === Setup Functions ===
    def _afinn_setup(self): self.client = Afinn()
    def _VADER_setup(self): self.client = SentimentIntensityAnalyzer()
    def _FLAIR_setup(self): self.client = Classifier.load('sentiment')
    def _perpective_setup(self):
        self.client = build("commentanalyzer", "v1alpha1",
            developerKey=os.getenv('PERSPECTIVE_API_KEY'),
            discoveryServiceUrl="https://commentanalyzer.googleapis.com/$discovery/rest?version=v1alpha1",
            static_discovery=False)
    def _regard_setup(self): self.client = load(EVALUATION_MEASUREMENT_PATH+"regard", module_type="measurement")
    def _log_likelyhood_setup(self): self.client = load("perplexity", module_type="metric")
    
    # === Score Functions ===
    def _get_Afinn_scores(self, sentence = False):
        if sentence:
            self.evaluation_file[self.key] = [self.client.score(pred) for pred in self.unmarked_sentence_list]
        self.evaluation_file[self.key] = [self.client.score(pred) for pred in self.predictions_list]
    
    def _get_VADER_scores(self, sentence = False):
        if sentence:
            self.evaluation_file[self.key] = [round(self.client.polarity_scores(pred)['compound'], 2) for pred in self.unmarked_sentence_list]
        self.evaluation_file[self.key] = [round(self.client.polarity_scores(pred)['compound'], 2) for pred in self.predictions_list]
    
    def _get_FLAIR_scores(self, sentence = False):
        if sentence:
            self.evaluation_file[self.key] = [self.client.predict(Sentence(pred)).tag for pred in self.unmarked_sentence_list]
        self.evaluation_file[self.key] = [self.client.predict(Sentence(pred)).tag for pred in self.predictions_list]
    
    def _get_TextBlob_scores(self, sentence = False):
        if sentence:
            self.evaluation_file[self.key] = [TextBlob(pred).sentences[0].sentiment.polarity for pred in self.unmarked_sentence_list]
        self.evaluation_file[self.key] = [TextBlob(pred).sentences[0].sentiment.polarity for pred in self.predictions_list]
    
    def _get_regard_scores(self): 
        try:
            scores = [item for item in self.client.compute(data=self.xyz_subject)['regard']]
            scores_df = {cat: [] for cat in REGARD_CATEGORIES}
            for item in scores:
                for entry in item:
                    if str(entry["label"]).capitalize() in scores_df:
                        try: 
                            scores_df[str(entry["label"]).capitalize()].append(entry["score"])
                        except Exception as e: 
                            logger.error("extractRegardScores: "+str(e))
            for cat in REGARD_CATEGORIES:
                self.evaluation_file[f"{REGARD} {cat}"] = scores_df[cat]
            return False
        except Exception as e:
            logger.error("_get_regard_scores: "+str(e))
            return True
    
    def _get_perspective_scores(self):
        try:
            scores = [self._perspective_request(s) for s in tqdm(self.unmarked_sentence_list, total=len(self.unmarked_sentence_list))]
            scores_df = {cat: [] for cat in PERSPECTIVE_CATEGORIES}
            for item in scores:
                for key in item:
                    scores_df[key].append(item[key])
            for cat in PERSPECTIVE_CATEGORIES:
                self.evaluation_file[f"{PERSPECTIVE} {cat}"] = scores_df[cat]
            return False
        except Exception as e:
            logger.error("_get_perspective_scores: "+str(e))
            return True

    def _perspective_request(self, sentence):
        row, timeError = {}, 0
        while timeError < 20000:
            try:
                response = self.client.comments().analyze(body={
                    'comment': {'text': sentence},
                    'requestedAttributes': {cat: {} for cat in PERSPECTIVE_CATEGORIES}
                }).execute()
                #sleep(0.9)
                for cat in PERSPECTIVE_CATEGORIES:
                    score = response['attributeScores'][cat]['summaryScore']['value']
                    row[cat] = score #row.get(cat, 0) + 1 if score >= 0.5 else row.get(cat, 0)
                return row
            except HttpError as e:
                if e.resp.status in [400, 429]:
                    if e.resp.status == 429: time.sleep(0.2)
                    timeError += 1
                    logger.warning(f"{timeError} - Perspective Request Time")
            except Exception as e:
                logger.error("getPerplexityScores: "+str(e))
                timeError += 1
        return {cat: row.get(cat, 0) for cat in PERSPECTIVE_CATEGORIES}
    
    def _get_log_likelyhood_scores(self):
        scores = []
        for idx in tqdm(range(len(self.unmarked_sentence_list)), total=len(self.unmarked_sentence_list)):
            scores.append(self.get_log_probability_for_word(self.unmarked_sentence_list[idx], self.predictions_list[idx]))
        self.evaluation_file[self.key] = scores
    
    
    def get_log_probability_for_word(self, sentence, target_word):
        model_name = 'bert-base-uncased'
        tokenizer = BertTokenizer.from_pretrained(model_name)
        model = BertForMaskedLM.from_pretrained(model_name)
        model.eval()

        # Tokenize target word
        target_tokens = tokenizer.tokenize(target_word)
        num_masks = len(target_tokens)

        # Add appropriate number of [MASK] tokens
        masked_sentence = sentence + " " + " ".join([MASKBERT] * num_masks)
        inputs = tokenizer(masked_sentence, return_tensors="pt")
        mask_indices = (inputs['input_ids'][0] == tokenizer.mask_token_id).nonzero(as_tuple=True)[0]

        if len(mask_indices) != num_masks:
            print(f"⚠️ Number of [MASK] tokens doesn't match target token length for '{target_word}'")
            return None

        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits

        log_probs = torch.log_softmax(logits, dim=-1)
        target_ids = tokenizer.convert_tokens_to_ids(target_tokens)

        # Sum log probabilities of each masked token prediction
        total_log_prob = 0.0
        for i, mask_idx in enumerate(mask_indices):
            token_id = target_ids[i]
            token_log_prob = log_probs[0, mask_idx, token_id].item()
            total_log_prob += token_log_prob

        return total_log_prob