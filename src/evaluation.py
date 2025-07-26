# === Imports ===
from src.lib import *
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
import spacy
logging.set_verbosity_error()
warnings.filterwarnings('ignore')

# === Constants ===
EVALUATION_MEASUREMENT_PATH = '.venv/evaluate/measurements/'

class Evaluation:
    
    def __init__(self):
        self.template_file = pd.read_csv(PATH_DATASET + 'templates.csv')
        
        self.initialize_tools = {
            # AFINN: self._afinn_setup,
            VADER: self._VADER_setup,
            # FLAIR: self._FLAIR_setup,
            REGARD: self._regard_setup,
            PERSPECTIVE: self._perpective_setup,
            POS: self._pos_setup,
        }
        
        self.tool_functions = {
            # AFINN: self._get_Afinn_scores,
            VADER: self._get_VADER_scores,
            # FLAIR: self._get_FLAIR_scores,
            # TEXTBLOB: self._get_TextBlob_scores,
            REGARD: self._get_regard_scores,
            PERSPECTIVE: self._get_perspective_scores,
            POS: self._get_POS_scores,
        }
    
    def evaluate(self, model_name, prompt_num = PROMPT_DEFAULT):
        self.model_name = model_name
        self.prompt_num = prompt_num
        os.makedirs(f"{PATH_EVALUATIONS}prompt_{self.prompt_num}/", exist_ok=True)
        self.df_to_check_list = self._get_evaluation_file()
        if self.df_to_check_list[0].empty and self.df_to_check_list[1].empty: #There is an error
            return True 
        for _, df in enumerate(self.df_to_check_list):
            if df.empty:
                continue
            self.df_to_check = df
            self.template_list = [row[TEMPLATE] for _, row in self.df_to_check.iterrows()]
            self.predictions_list = [str(row[PREDICTION]) for _, row in self.df_to_check.iterrows()]
            self.unmarked_sentence_list = [f"{row[UNMARKED]} {row[PREDICTION]}" for _, row in self.df_to_check.iterrows()]
            self.xyz_subject = [f"{re.sub('The '+SUBJECT_, 'xyz', row[TEMPLATE])} {row[PREDICTION]}." for _, row in self.df_to_check.iterrows()]
                    
            start_evaluation = True
            for key, score_function in self.tool_functions.items():
                if key in self.initialize_tools:
                    self.initialize_tools[key]()
                self.key = key
                if key == REGARD and not any(f"{key} {cat}" in self.df_to_check.columns for cat in REGARD_CATEGORIES):
                    if start_evaluation: logger.info(f"📊 Evaluating {model_name} [prompt {prompt_num}]"); start_evaluation = False
                    logger.info(f"  🧮 Calculating {key} scores...")
                    res = score_function()
                    if res: break
                    self.save_csv()
                elif key == PERSPECTIVE and not any(f"{key} {cat}" in self.df_to_check.columns for cat in PERSPECTIVE_CATEGORIES):
                    if start_evaluation: logger.info(f"📊 Evaluating {model_name} [prompt {prompt_num}]"); start_evaluation = False
                    logger.info(f"  🧮 Calculating {key} scores...")
                    res = score_function()
                    if res: break
                    self.save_csv()
                elif key != REGARD and key != PERSPECTIVE and key not in self.df_to_check.columns:
                    if start_evaluation: logger.info(f"📊 Evaluating {model_name} [prompt {prompt_num}]"); start_evaluation = False
                    logger.info(f"  🧮 Calculating {key} scores...")
                    score_function()
                    self.save_csv()
                    
                
        logger.info(f"✅ {MODELS_LABELS[model_name]} [prompt {int(self.prompt_num)}]")
        return False
        

    def _check_evaluation_file_integrity(self, evaluation_file):
        for row_idx, row in evaluation_file.iterrows():
            for col in evaluation_file.columns:
                value = row[col]
                if pd.isna(value) or (isinstance(value, str) and value.strip() == ""):
                    #if col != PREDICTION:
                    logger.info(f"⚠️ {MODELS_LABELS[self.model_name]} [prompt {int(self.prompt_num)}] invalid cell [{row_idx} - {col}]")    
                    return row_idx
        return evaluation_file.shape[0]
        
    def _get_evaluation_file(self):
        prediction_file = f"{PATH_GENERATIONS}prompt_{self.prompt_num}/{self.model_name}.csv"
        evaluation_file = f"{PATH_EVALUATIONS}prompt_{self.prompt_num}/{self.model_name}.csv"
        
        if os.path.exists(prediction_file): 
            prediction_file = pd.read_csv(prediction_file)
            if os.path.exists(evaluation_file): 
                evaluation_file = pd.read_csv(evaluation_file)
                num_sample_evaluated = self._check_evaluation_file_integrity(evaluation_file) #Check weather there are empty cells
                
                if evaluation_file.shape[0] > prediction_file.shape[0]:
                    logger.info(f"⚠️ {MODELS_LABELS[self.model_name]} [prompt {int(self.prompt_num)}] evaluation file bigger than generation file")    
                    return [pd.DataFrame(), pd.DataFrame()]
                elif num_sample_evaluated < prediction_file.shape[0]:
                    if num_sample_evaluated == 0:
                        return [pd.DataFrame(), prediction_file]
                    logger.info(f"🔙 {num_sample_evaluated} sentences imported")
                    return [evaluation_file[:num_sample_evaluated], prediction_file[num_sample_evaluated:]]
                return[evaluation_file, pd.DataFrame()] #Evaluation file already completed, needs to be checked
            else:    
                return [pd.DataFrame(), prediction_file]
        else: #If pred does not exist ERROR
            logger.warning(f"⚠️ {self.model_name} [prompt {self.prompt_num}] prediction file not found ⚠️")
            return [pd.DataFrame(), pd.DataFrame()]
    
    def _get_expected_word(self, sentence):
            for _, row in self.template_file.iterrows():
                if sentence.lower().strip().startswith(row[TEMPLATE].lower()):
                    return row[EXPECTED_WORD_TYPE].split()

    def save_csv(self):
        df = pd.concat(self.df_to_check_list)
        if self.model_name in MODEL_MLM:
            for idx, _ in enumerate(PROMPTS):
                df.to_csv(
                    f"{PATH_EVALUATIONS}prompt_{self.prompt_num}/{self.model_name}.csv",
                    index=False
                )
        else:
            df.to_csv(
                f"{PATH_EVALUATIONS}prompt_{self.prompt_num}/{self.model_name}.csv",
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
    def _pos_setup(self): self.client = spacy.load("en_core_web_sm")
    
    def _get_POS_scores(self):
        noun_tags = {"NN", "NNS", "NNP", "NNPS"}
        verb_tags = {"VB", "VBD", "VBG", "VBN", "VBP", "VBZ"}
        POS_scores = []
        for idx, sentence in tqdm(enumerate(self.unmarked_sentence_list), total=len(self.unmarked_sentence_list)):
            sentence = self.client(sentence)
            sentence_tokens = [token for token in sentence if not token.is_space and not token.is_punct]
            last_token = sentence[-1]
            ok_types = self._get_expected_word(self.template_list[idx])
            
            if last_token.tag_ in noun_tags and "NOUN" in ok_types:
                POS_scores.append(True)
            elif last_token.tag_ in verb_tags and "VERB" in ok_types:
                POS_scores.append(True)
            else:
                POS_scores.append(False)
        self.df_to_check[self.key] = POS_scores
    
    
    # === Score Functions ===
    def _get_Afinn_scores(self, sentence = False):
        if sentence:
            self.df_to_check[self.key] = [self.client.score(pred) for pred in self.unmarked_sentence_list]
        self.df_to_check[self.key] = [self.client.score(pred) for pred in self.predictions_list]
    
    def _get_VADER_scores(self, sentence = False):
        if sentence:
            self.df_to_check[self.key] = [round(self.client.polarity_scores(pred)['compound'], 2) for pred in self.unmarked_sentence_list]
        self.df_to_check[self.key] = [round(self.client.polarity_scores(pred)['compound'], 2) for pred in self.predictions_list]
    
    def _get_FLAIR_scores(self, sentence = False):
        if sentence:
            self.df_to_check[self.key] = [self.client.predict(Sentence(pred)).tag for pred in self.unmarked_sentence_list]
        self.df_to_check[self.key] = [self.client.predict(Sentence(pred)).tag for pred in self.predictions_list]
    
    def _get_TextBlob_scores(self, sentence = False):
        if sentence:
            self.df_to_check[self.key] = [TextBlob(pred).sentences[0].sentiment.polarity for pred in self.unmarked_sentence_list]
        self.df_to_check[self.key] = [TextBlob(pred).sentences[0].sentiment.polarity for pred in self.predictions_list]
    
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
                self.df_to_check[f"{REGARD} {cat}"] = scores_df[cat]
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
                self.df_to_check[f"{PERSPECTIVE} {cat}"] = scores_df[cat]
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
    