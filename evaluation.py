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
warnings.filterwarnings('ignore')

# === Constants ===
EVALUATION_MEASUREMENT_PATH = '.venv/evaluate/measurements/'

class Evaluation:
    
    def __init__(self):
        os.makedirs(OUTPUT_EVALUATION + self.prompt_num+"/", exist_ok=True)
        self.initialize_tools = {
            AFINN: self._afinn_setup,
            VADER: self._VADER_setup,
            FLAIR: self._FLAIR_setup,
            REGARD: self._regard_setup,
            PERSPECTIVE: self._perpective_setup,
        }
        
        self.tool_functions = {
            AFINN: self._get_Afinn_scores,
            VADER: self._get_VADER_scores,
            FLAIR: self._get_FLAIR_scores,
            TEXTBLOB: self._get_TextBlob_scores,
            REGARD: self._get_regard_scores,
            PERSPECTIVE: self._get_perspective_scores,
        }
        
    def evaluate(self, model_name, prompt_num):
        self.model_name = model_name
        self.prompt_num = prompt_num
        logger.info(f"○ Evaluating {model_name} with {prompt_num}")
        
        self.evaluation_file = self._get_template_file()
        if self.evaluation_file == None:
            return None
        
        self.predictions_list = [str(row[PREDICTION]) for _, row in self.evaluation_file.iterrows()]
        self.unmarked_sentence_list = [f"{row[UNMARKED]} {row[PREDICTION]}" for _, row in self.evaluation_file.iterrows()]
        self.xyz_subject = [f"{re.sub(SUBJECT_, 'xyz', row[TEMPLATE])} {row[PREDICTION]}." for _, row in self.evaluation_file.iterrows()]
        
        for key, score_function in self.tool_functions.items():
            self.client = self.initialize_tools.get(key, lambda: "")()
            self.key = key
            logger.info(f"○ Calculating {key} scores...")
            
            if key == REGARD and not any(f"{key} {cat}" in self.evaluation_file.columns for cat in REGARD_CATEGORIES):
                    res = score_function()
                    if res == None: break
            elif key == PERSPECTIVE and not any(f"{key} {cat}" in self.evaluation_file.columns for cat in PERSPECTIVE_CATEGORIES):
                    res = score_function()
                    if res == None: break
            elif key not in self.evaluation_file.columns:
                score_function()
        
            self.evaluation_file.to_csv(OUTPUT_EVALUATION + self.prompt_num + "/" + self.model_name+".csv", index=False)
        logger.info(f"○ {model_name} with {self.prompt_num} evaluated!")    
        
    def _get_template_file(self):
        prediction_file = OUTPUT_SENTENCES + self.prompt_num + "/" + self.model_name+".csv"
        evaluation_file = OUTPUT_EVALUATION + self.prompt_num + "/" + self.model_name+".csv"

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
            return None
    
    # === Setup Functions ===
    def _afinn_setup(self): return Afinn()
    def _VADER_setup(self): return SentimentIntensityAnalyzer()
    def _FLAIR_setup(self): return Classifier.load('sentiment')
    def _perpective_setup(self):
        return build("commentanalyzer", "v1alpha1",
            developerKey=os.getenv('PERSPECTIVE_API_KEY'),
            discoveryServiceUrl="https://commentanalyzer.googleapis.com/$discovery/rest?version=v1alpha1",
            static_discovery=False)
    def _regard_setup(self): return load(EVALUATION_MEASUREMENT_PATH+"regard", module_type="measurement")

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
        except Exception as e:
            logger.error("_get_regard_scores: "+str(e))
            return None
    
    def _get_perspective_scores(self):
        try:
            scores = [self._perspective_request(self.client, s) for s in tqdm(self.unmarked_sentence_list, total=self.unmarked_sentence_list.shape[0])]
            scores_df = {cat: [] for cat in PERSPECTIVE_CATEGORIES}
            for item in scores:
                for key in item:
                    scores_df[key].append(item[key])
            for cat in PERSPECTIVE_CATEGORIES:
                self.evaluation_file[f"{PERSPECTIVE} {cat}"] = scores_df[cat]
        except Exception as e:
            logger.error("_get_perspective_scores: "+str(e))
            return None
    
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
                