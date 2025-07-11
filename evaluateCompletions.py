# === Imports ===
from lib import *
from time import sleep
from afinn import Afinn
from evaluate import load 
import warnings, os, re, time
#from google.cloud import language_v2
from googleapiclient import discovery
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from flair.nn import Classifier
from flair.data import Sentence
from textblob import TextBlob
warnings.filterwarnings('ignore')

# VADER_sent = 'VADER_sent'
# AFINN_sent = 'AFINN_sent'
# FLAIR_sent = 'Flair_sent'
# TEXTBLOB_sent = "TextBlob_sent"

# === Constants ===
EVALUATION_MEASUREMENT_PATH = '.venv/evaluate/measurements/'
EVALUATION_METRICS_PATH = '.venv/evaluate/metrics/'

# === Setup Functions ===
#def afinnSetup(): return Afinn()
def vaderSetup(): return SentimentIntensityAnalyzer()
def afinnSetup(): return Afinn()
def flairSetup(): return Classifier.load('sentiment')
def perspectiveSetup():
    return build("commentanalyzer", "v1alpha1",
        developerKey=os.getenv('PERSPECTIVE_API_KEY'),
        discoveryServiceUrl="https://commentanalyzer.googleapis.com/$discovery/rest?version=v1alpha1",
        static_discovery=False)
def regardSetup(): return load(EVALUATION_MEASUREMENT_PATH+"regard", module_type="measurement")
#def googleCloudNLSetup(): return language_v2.LanguageServiceClient()

initialize_tools = {
    AFINN: afinnSetup,
    VADER: vaderSetup,
    FLAIR: flairSetup,
    # AFINN_sent: afinnSetup,
    # VADER_sent: vaderSetup,
    # FLAIR_sent: flairSetup,
    REGARD: regardSetup,
    PERSPECTIVE: perspectiveSetup,
}

# === Utility Functions ===
def find_subsequence_indices(s1, s2):
    s1 = s1.replace(r".","")
    words_s1, words_s2 = s1.split(), s2.split()
    len_s2 = len(words_s2)
    for i in range(len(words_s1) - len_s2 + 1):
        if words_s1[i:i + len_s2] == words_s2:
            return i, i + len_s2
    return None

def extractRegardScores(data):
    scores = {cat: [] for cat in REGARD_CATEGORIES}
    for item in data:
        for entry in item:
            if str(entry["label"]).capitalize() in scores:
                try: scores[str(entry["label"]).capitalize()].append(entry["score"])
                except Exception as e: logger.error("extractRegardScores: "+str(e))
    return scores

def extractPerspectiveScores(data):
    scores = {cat: [] for cat in PERSPECTIVE_CATEGORIES}
    for item in data:
        for key in item:
            scores[key].append(item[key])
    return scores

# def callGoogleCloudSentimentAnalisysScores(pred, templateFile, client):
#     lst = []
#     for sentence in tqdm(pred, total=templateFile.shape[0]):
#         try:
#             response = client.analyze_sentiment(request={
#                 "document": {
#                     "content": sentence,
#                     "type_": language_v2.Document.Type.PLAIN_TEXT,
#                     "language_code": "en"
#                 }, "encoding_type": language_v2.EncodingType.UTF8
#             })
#             lst.append(round(response.document_sentiment.score, 2))
#         except Exception as e:
#             logger.error("GoogleCloudSentimentAnalisysScores: "+str(e))
#             lst.append(0)
#     return lst

def perspectiveRequest(client, sentence):
    row, timeError = {}, 0
    while timeError < 20000:
        try:
            response = client.comments().analyze(body={
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

# === Score Functions ===
def getAfinnScores(df, client):
    logger.info("○ Calculating Afinn score...")
    return [client.score(str(row[PREDICTION])) for _, row in df.iterrows()]

# def getAfinnScores_sent(df, client):
#     logger.info("○ Calculating Afinn score...")
#     sentences = [f"{row[UNMARKED]} {row[PREDICTION]}" for _, row in df.iterrows()]
#     return [client.score(str(sent)) for sent in sentences]

def getFlairScores(df, client):
    logger.info("○ Calculating Flair score...")
    score = []
    for _, row in df.iterrows():
        sentence = Sentence(row[PREDICTION])
        client.predict(sentence)
        score.append(sentence.tag)
    return score

# def getFlairScores_sent(df, client):
#     logger.info("○ Calculating Flair score...")
#     score = []
#     sentences = [f"{row[UNMARKED]} {row[PREDICTION]}" for _, row in df.iterrows()]
#     for sent in sentences:
#         sentence = Sentence(sent)
#         client.predict(sentence)
#         score.append(sentence.tag)
#     return score

def getTextBlobScores(df, client):
    logger.info("○ Calculating TextBlob score...")
    score = []
    for _, row in df.iterrows():
        score.append(TextBlob(row[PREDICTION]).sentences[0].sentiment.polarity)
    return score

# def getTextBlobScores_sent(df, client):
#     logger.info("○ Calculating TextBlob score...")
#     score = []
#     sentences = [f"{row[UNMARKED]} {row[PREDICTION]}" for _, row in df.iterrows()]
#     for sent in sentences:
#         score.append(TextBlob(sent).sentences[0].sentiment.polarity)
#     return score

def getVaderScores(df, client):
    logger.info("○ Calculating VADER score...")
    return [round(client.polarity_scores(str(row[PREDICTION]))['compound'], 2) for _, row in df.iterrows()]
    
# def getVaderScores_sent(df, client):
#     logger.info("○ Calculating VADER score...")
#     sentences = [f"{row[UNMARKED]} {row[PREDICTION]}" for _, row in df.iterrows()]
#     return [round(client.polarity_scores(str(sent))['compound'], 2) for sent in sentences]

# def getGoogleCloudSentimentAnalisysScores(df, client):
#     logger.info("○ Calculating Google Cloud Sentiment score...")
#     pred = [str(row[PREDICTION]) for _, row in df.iterrows()]
#     return callGoogleCloudSentimentAnalisysScores(pred, df, client)

def getRegardScore(df, client):
    logger.info("○ Calculating Regard score...")
    sentences = [f"{re.sub(SUBJECT_, 'xyz', row[TEMPLATE])} {row[PREDICTION]}." for _, row in df.iterrows()]
    try:
        return [item for item in client.compute(data=sentences)['regard']]
    except Exception as e:
        logger.error("getRegardScore: "+str(e))

def getPerspectiveScore(df, client):
    logger.info("○ Calculating Perspective score...")
    sentences = [f"{row[UNMARKED]} {row[PREDICTION]}" for _, row in df.iterrows()]
    ret = [perspectiveRequest(client, s) for s in tqdm(sentences, total=df.shape[0])]
    return ret

score_functions = {
    AFINN: getAfinnScores,
    VADER: getVaderScores,
    FLAIR: getFlairScores,
    TEXTBLOB: getTextBlobScores,
    # AFINN_sent: getAfinnScores_sent,
    # VADER_sent: getVaderScores_sent,
    # FLAIR_sent: getFlairScores_sent,
    # TEXTBLOB_sent: getTextBlobScores_sent,
    REGARD: getRegardScore,
    PERSPECTIVE: getPerspectiveScore,
}

# === Main Evaluation Function ===
def evaluatePrediction(modelList):
    inputFolder, outputFolder = OUTPUT_SENTENCES, OUTPUT_EVALUATION
    os.makedirs(outputFolder, exist_ok=True)
    
    for modelName in modelList:
        logger.info(f"○ Evaluating {modelName}")
        preTemplateFile, templateFile = getTemplateFile(modelName, inputFolder, outputFolder)
        if not isinstance(preTemplateFile, pd.DataFrame):
            logger.warning(f"○ {modelName} prediction file not found")
            break
        templateFile = pd.concat([preTemplateFile, templateFile]) if not preTemplateFile.empty else templateFile
        
        evaluation_success = True
        for key, func in score_functions.items():
            client = initialize_tools.get(key, lambda: "")()
            if key == REGARD:
                if not any(f"{key} {cat}" in templateFile.columns for cat in REGARD_CATEGORIES):
                    result = func(templateFile, client)
                    if result is None:
                        logger.error("Regard function returned None")
                        evaluation_success = False
                        break
                    scores = extractRegardScores(result)
                    for cat in REGARD_CATEGORIES:
                        templateFile[f"{REGARD} {cat}"] = scores[cat]
            elif key == PERSPECTIVE:
                if not any(f"{key} {cat}" in templateFile.columns for cat in PERSPECTIVE_CATEGORIES):
                    scores = extractPerspectiveScores(func(templateFile, client))
                    for cat in PERSPECTIVE_CATEGORIES:
                        templateFile[f"{PERSPECTIVE} {cat}"] = scores[cat]
            elif key not in templateFile.columns:
                templateFile[key] = func(templateFile, client)
        
        templateFile.to_csv(f"{outputFolder}{modelName}.csv", index=False)
        logger.info(f"○ {modelName} {'OK!' if evaluation_success else 'FAILED!'}")

# === Run Evaluation ===
evaluatePrediction(MODEL_LIST_FULL)