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
import ast
logging.set_verbosity_error()
warnings.filterwarnings('ignore')

# === Constants ===
EVALUATION_MEASUREMENT_PATH = '.venv/evaluate/measurements/'

class Evaluation:
    
    def __init__(self, model_name, num_predictions = 1):
        self.model_name = model_name
        self.num_predictions = num_predictions
        self.template_expected_type = pd.read_csv(TEMPLATE_PATH)

        
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
    
    def evaluate(self, prompt_num = PROMPT_DEFAULT):
        self.prompt_num = prompt_num        
        self.path_generations = PATH_GENERATIONS+"/" if self.num_predictions == 1 else PATH_GENERATIONS +'_top_'+str(self.num_predictions)+"/"
        self.path_evaluations = PATH_EVALUATIONS+"/" if self.num_predictions == 1 else PATH_EVALUATIONS +'_top_'+str(self.num_predictions)+"/"
        
        self.prediction_file = f"{self.path_generations}prompt_{self.prompt_num}/{self.model_name}.csv" if self.num_predictions == 1 else f"{self.path_generations+self.model_name}.csv"
        self.evaluation_file = f"{self.path_evaluations}prompt_{self.prompt_num}/{self.model_name}.csv" if self.num_predictions == 1 else f"{self.path_evaluations+self.model_name}.csv"
        
        if not os.path.exists(f"{self.path_generations}"):
            logger.warning(f"⚠️ {self.model_name} [prompt {self.prompt_num}] prediction file not found  in {self.path_evaluation} ⚠️")
            return False
            
        if self.num_predictions == 1 and prompt_num != 0 and (self.model_name == BERT_BASE or self.model_name == BERT_LARGE or self.model_name == ROBERTA_BASE or self.model_name == ROBERTA_LARGE):
            self.copy_file(f"{PATH_EVALUATIONS}prompt_0/{self.model_name}.csv", f"{PATH_EVALUATIONS}/prompt_{self.prompt_num}/{self.model_name}.csv")
            logger.info(f"✅ {MODELS_LABELS[self.model_name]} [prompt {self.prompt_num}]")
            return False
        
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
                    
            start_evaluation = True
            for key, score_function in self.tool_functions.items():
                if key in self.initialize_tools:
                    err = self.initialize_tools[key]()
                    if err:
                        break
                self.key = key
                if key == REGARD and not any(f"{key} {cat}" in self.df_to_check.columns for cat in REGARD_CATEGORIES):
                    if start_evaluation: logger.info(f"📊 Evaluating {self.model_name} [prompt {prompt_num}]"); start_evaluation = False
                    logger.info(f"  🧮 Calculating {key} scores...")
                    res = score_function()
                    if res: break
                    self.save_csv()
                elif key == PERSPECTIVE and not any(f"{key} {cat}" in self.df_to_check.columns for cat in PERSPECTIVE_CATEGORIES):
                    if start_evaluation: 
                        logger.info(f"📊 Evaluating {self.model_name} [prompt {prompt_num}]"); start_evaluation = False
                    logger.info(f"  🧮 Calculating {key} scores...")
                    res = score_function()
                    if res: break
                    self.save_csv()
                elif key != REGARD and key != PERSPECTIVE and key not in self.df_to_check.columns:
                    if start_evaluation: logger.info(f"📊 Evaluating {self.model_name} [prompt {prompt_num}]"); start_evaluation = False
                    logger.info(f"  🧮 Calculating {key} scores...")
                    score_function()
                    self.save_csv()
                    
        if not (err and res): #No Error
            logger.info(f"✅ {MODELS_LABELS[self.model_name]} [prompt {int(self.prompt_num)}]")
            return False
        return True #Error

    def _check_evaluation_file_integrity(self, evaluation_file):
        for row_idx, row in evaluation_file.iterrows():
            for col in evaluation_file.columns:
                value = str(row[col])
                if pd.isna(value) or (isinstance(value, str) and value.strip() == ""):
                    if col != PREDICTION:
                        logger.info(f"⚠️ {MODELS_LABELS[self.model_name]} [prompt {int(self.prompt_num)}] invalid cell [{row_idx} - {col}]")    
                        return row_idx
        return evaluation_file.shape[0]
        
    def _get_evaluation_file(self):
        if os.path.exists(self.prediction_file): 
            prediction_file = pd.read_csv(self.prediction_file)
            if os.path.exists(self.evaluation_file): 
                evaluation_file = pd.read_csv(self.evaluation_file)
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
            for _, row in self.template_expected_type.iterrows():
                if sentence.lower().strip().startswith(row[TEMPLATE].lower()):
                    return row[EXPECTED_WORD_TYPE].split()

    def save_csv(self):
        df = pd.concat(self.df_to_check_list)
        
        if self.num_predictions == 1:
            os.makedirs(f"{self.path_evaluations}/prompt_{self.prompt_num}/", exist_ok=True)
            if self.model_name in MODEL_MLM:
                for idx, _ in enumerate(PROMPTS):
                    df.to_csv(f"{self.path_evaluations}/prompt_{self.prompt_num}/{self.model_name}.csv", index=False)
            else:
                df.to_csv(f"{self.path_evaluations}/prompt_{self.prompt_num}/{self.model_name}.csv", index=False)
        else:
            os.makedirs(self.path_evaluations, exist_ok=True)      
            df.to_csv(self.evaluation_file, index=False)  
            
    # === Setup Functions ===
    def _afinn_setup(self): self.client = Afinn(); return False
    def _VADER_setup(self): self.client = SentimentIntensityAnalyzer(); return False
    def _FLAIR_setup(self): self.client = Classifier.load('sentiment'); return False
    def _perpective_setup(self):
        api_key = os.getenv('PERSPECTIVE_API_KEY')
        if api_key is None or api_key == "":
            logger.error(f"⚠️ PERSPECTIVE_API_KEY is missing")
            return True
        self.client = build("commentanalyzer", "v1alpha1",
            developerKey=api_key,
            discoveryServiceUrl="https://commentanalyzer.googleapis.com/$discovery/rest?version=v1alpha1",
            static_discovery=False)
        return False
    def _regard_setup(self): self.client = load(EVALUATION_MEASUREMENT_PATH+"regard", module_type="measurement"); return False
    def _pos_setup(self): self.client = spacy.load("en_core_web_sm"); return False
    
    def _get_POS_scores(self):
        noun_tags = {"NN", "NNS", "NNP", "NNPS"}
        verb_tags = {"VB", "VBD", "VBG", "VBN", "VBP", "VBZ"}
        POS_scores = []

        try:
            for idx, row in tqdm(self.df_to_check.iterrows(), total=len(self.df_to_check)):

                base_sentence = row[UNMARKED]
                preds = row[PREDICTION]
                # Build sentences
                if self.num_predictions == 1:
                    sentences = [f"{base_sentence} {preds}"]
                else:
                    sentences = [f"{base_sentence} {p}" for p in ast.literal_eval(preds)]

                # Expected POS types for this row
                ok_types = self._get_expected_word(self.template_list[idx])

                # Track validity of all predictions for this row
                all_correct = True

                for sent in sentences:
                    doc = self.client(sent)
                    last_token = doc[-1]

                    if last_token.tag_ in noun_tags and "NOUN" in ok_types:
                        continue  # good
                    elif last_token.tag_ in verb_tags and "VERB" in ok_types:
                        continue  # good
                    else:
                        all_correct = False
                        break  # no need to check the remaining predictions

                POS_scores.append(all_correct)

            self.df_to_check[self.key] = POS_scores
            return False  # function executed correctly

        except Exception as e:
            logger.error("_get_POS_scores: " + str(e))
            return True
    
    
    # === Score Functions ===
    def _get_Afinn_scores(self, sentence = False):
        if sentence:
            self.df_to_check[self.key] = [self.client.score(pred) for pred in self.unmarked_sentence_list]
        self.df_to_check[self.key] = [self.client.score(pred) for pred in self.predictions_list]
    
    def _get_VADER_scores(self, sentence = False):
        if self.num_predictions == 1:
            if sentence:
                self.df_to_check[self.key] = [round(self.client.polarity_scores(pred)['compound'], 2) for pred in self.unmarked_sentence_list]
            self.df_to_check[self.key] = [round(self.client.polarity_scores(pred)['compound'], 2) for pred in self.predictions_list]
        else:
            self.df_to_check[self.key] = [
                np.average([
                    round(self.client.polarity_scores(pred)['compound'], 2)
                    for pred in ast.literal_eval(list_pred)  # convert string to list
                ])
                for list_pred in self.predictions_list
            ]
            
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
            # Prepare empty columns
            for cat in REGARD_CATEGORIES:
                self.df_to_check[f"{REGARD} {cat}"] = np.nan

            # Process row by row
            for idx, row in tqdm(self.df_to_check.iterrows(), total=self.df_to_check.shape[0]):
                # Build sentences for this row
                if self.num_predictions == 1:
                    sentences = [f"{re.sub("The " + SUBJECT_, "xyz", row[TEMPLATE])} {row[PREDICTION]}."]
                else:
                    sentences = [f"{re.sub("The " + SUBJECT_, "xyz", row[TEMPLATE])} {p}." for p in ast.literal_eval(row[PREDICTION])]

                # Compute regard for this row's sentences
                scores = self.client.compute(data=sentences)["regard"]

                # Prepare accumulator for this row
                row_cat_scores = {cat: [] for cat in REGARD_CATEGORIES}

                # Loop over each sentence produced for this row
                for sentence_scores in scores:
                    for entry in sentence_scores:
                        label = str(entry["label"]).capitalize()
                        if label in row_cat_scores:
                            row_cat_scores[label].append(entry["score"])

                # Average per category (within this row only)
                for cat in REGARD_CATEGORIES:
                    vals = row_cat_scores[cat]
                    avg_val = np.mean(vals) if len(vals) > 0 else 0
                    self.df_to_check.loc[idx, f"{REGARD} {cat}"] = avg_val

            return False

        except Exception as e:
            logger.error("_get_regard_scores: " + str(e))
            return True
    

    def _get_perspective_scores(self):
        try:
            # Prepare output columns
            for cat in PERSPECTIVE_CATEGORIES:
                self.df_to_check[f"{PERSPECTIVE} {cat}"] = np.nan

            # Iterate row by row
            for idx, row in tqdm(self.df_to_check.iterrows(), total=len(self.df_to_check)):

                # Build sentences for this row
                if self.num_predictions == 1:
                    sentences = [f"{row[UNMARKED]} {row[PREDICTION]}"]
                else:
                    sentences = [f"{row[UNMARKED]} {p}" for p in ast.literal_eval(row[PREDICTION])]

                # Accumulator for this row
                row_scores = {cat: [] for cat in PERSPECTIVE_CATEGORIES}

                # Compute scores for each prediction sentence
                for sentence in sentences:
                    item = self._perspective_request(sentence)
                    for cat in PERSPECTIVE_CATEGORIES:
                        if cat in item:
                            row_scores[cat].append(item[cat])

                # Average scores for this row
                for cat in PERSPECTIVE_CATEGORIES:
                    values = row_scores[cat]
                    avg_val = np.mean(values) if len(values) > 0 else 0
                    self.df_to_check.loc[idx, f"{PERSPECTIVE} {cat}"] = avg_val

            # Add overall perspective score as mean across categories
            self.df_to_check[PERSPECTIVE] = self.df_to_check[
                [f"{PERSPECTIVE} {cat}" for cat in PERSPECTIVE_CATEGORIES]
            ].mean(axis=1)

            return False

        except Exception as e:
            logger.error("_get_perspective_scores: " + str(e))
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