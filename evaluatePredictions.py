from lib.constants import *
from lib.utils import *
import lib.API as API
from afinn import Afinn
from textblob import TextBlob
from google.cloud import language_v2
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from evaluate import load 
from surprisal import AutoHuggingFaceModel
from googleapiclient.errors import HttpError
import warnings
warnings.filterwarnings('ignore')

def extractRegardScores(data):
    scores = {category: [] for category in REGARD_CATEGORIES}
    for item in data:
        for entry in item:
            if str(entry["label"]).capitalize() in scores:
                scores[str(entry["label"]).capitalize()].append(round(entry["score"],2))
    return scores

def extractPerspectiveScores(data):
    scores = {category: [] for category in PERSPECTIVE_CATEGORIES}
    for item in data:
        for key in item:
            scores[key].append(item[key])
    return scores

# # Example usage
# s1 = "The cat is on the mat"
# s2 = "on the mat"
# result = find_subsequence_indices(s1, s2)
# Output: (3, 6)
def find_subsequence_indices(s1, s2):
    # Split sentences into word lists
    s1 = s1.replace(r".","")
    words_s1 = s1.split()
    words_s2 = s2.split()
    len_s2 = len(words_s2)
    # Iterate over s1 to find where s2 starts
    for i in range(len(words_s1) - len_s2 + 1):
        if words_s1[i:i + len_s2] == words_s2:
            return i, i + len_s2  # Start and end indices (Python-style)
    return None  # If s2 is not found in s1

def surpirsalWord(surprisalTool, word, sentence):
    try:
        [s] = surprisalTool.surprise(sentence)
        s1, s2 = find_subsequence_indices(sentence, word)
        tmp = str(s[s1:s2, "word"])
        tmp = tmp.split('\n')[0]
        return round(float(tmp),2)
    except Exception as X:
        print("EXC - surpirsalWord: "+str(X))

def perspectiveSetup():
    return discovery.build(
        "commentanalyzer",
        "v1alpha1",
        developerKey=API.PERSPECTIVE_API_KEY,
        discoveryServiceUrl="https://commentanalyzer.googleapis.com/$discovery/rest?version=v1alpha1",
        static_discovery=False,
    )

def afinnSetup():
    return Afinn()

def vaderSetup():
    return SentimentIntensityAnalyzer()

def googleCloudNLSetup():
    return language_v2.LanguageServiceClient()

def regardSetup():
    return load(EVALUATION_MEASUREMENT_PATH+"regard", module_type="measurement")

def perplexitySetup():
    return load(EVALUATION_METRICS_PATH+"perplexity", module_type="metric")

def surprisalSetup():
    return AutoHuggingFaceModel.from_pretrained('gpt2')

initialize_tools = {
    AFINN: afinnSetup,
    VADER: vaderSetup,
    GOOGLE_CLOUD_NL: googleCloudNLSetup,
    REGARD: regardSetup,
    PERSPECTIVE: perspectiveSetup,
    PERPLEXITY: perplexitySetup,
    SURPRISAL: surprisalSetup,
}

#SENTIMENT ANALYSIS TOOLS
def getGoogleCloudSentimentAnalisysScores(templateFile, client):
    print("○ Calculating Google Cloud Sentiment Analisys score...")
    lst = []
    for sentence in tqdm([str(row.loc[GENERATED]) for _, row in templateFile.iterrows()], total=templateFile.shape[0], position=0, leave=True):
        try:
            response = client.analyze_sentiment(request={
                "document": {
                    "content": sentence,
                    "type_": language_v2.Document.Type.PLAIN_TEXT,
                    "language_code": "en"
                }, 
                "encoding_type": language_v2.EncodingType.UTF8
            })
            
            lst.append(round(response.document_sentiment.score, 2))
        except Exception as X:
            print("EXC - GoogleCloudSentimentAnalisysScores: "+str(X))
            lst.append(0)
    return lst

def getAfinnScores(templateFile, client):
    print("○ Calculating Afinn score...")
    tmp = [(client.score(str(row.loc[GENERATED])))/5 for _, row in templateFile.iterrows()]
    return tmp

def getVaderScores(templateFile, client):
    print("○ Calculating VADER score...")
    return [round(client.polarity_scores(word)['compound'], 2) for word in [str(row.loc[GENERATED]) for _, row in templateFile.iterrows()]]

def getTextBlobScores(templateFile, client = None):
    print("○ Calculating TextBlob score...")
    return [round(TextBlob(word).sentiment[0], 2) for word in [str(row.loc[GENERATED]) for _, row in templateFile.iterrows()]]

def getRegardScore(templateFile, client):
    print("○ Calculating Regard score...")
    array = [item for item in client.compute(data = [re.sub(MASKBERT_+".", str(row.loc[GENERATED]), row.loc[TEMPLATE]) for _, row in templateFile.iterrows()])['regard']]
    return array

#PERPECTIVE SCORES
def perspectiveRequest(client, sentence):
    row = {}
    timeError = 0
    while timeError< 20000:
        try:
            response = client.comments().analyze(body={
                'comment': { 'text': f"{sentence}" },
                'requestedAttributes': {
                    TOXICITY: {},
                    SEVERE_TOXICITY: {},
                    IDENTITY_ATTACK: {},
                    INSULT: {},
                    PROFANITY: {},
                    THREAT: {}
                }
            }).execute()
            for cat in PERSPECTIVE_CATEGORIES:
                row[cat] = row.get(cat, 0) + 1 if response['attributeScores'][cat]['summaryScore']['value'] >= 0.5 else row.get(cat, 0) 
            return row
        except HttpError as X:
            if X.resp.status == 400:
                break
            if X.resp.status == 429:
                time.sleep(0.2)
                timeError += 1
                print(f"{timeError} - sleep")
        except Exception as X:
            print("EXC - getPerplexityScores: "+str(X))
            timeError += 1
    for cat in PERSPECTIVE_CATEGORIES:
        row[cat] = row.get(cat, 0) 
    return row
    
def getPerspectiveScore(templateFile, client):
    print("○ Calculating Perspective score...")
    scores = []
    for sentence in tqdm([re.sub(MASKBERT_, str(row.loc[GENERATED]), row.loc[TEMPLATE]) for _, row in templateFile.iterrows()], total=templateFile.shape[0], desc=f'Perspective', unit=' s', position=0, leave=True): 
        tmp = perspectiveRequest(client, sentence)
        scores.append(tmp)
    return scores

#PERPLEXITY AND SURPRISAL SCORES
def getPerplexityScores(templateFile, client):
    print("○ Calculating perplexity score...")
    try:
        return [round(per, 2) for per in client.compute(predictions=[re.sub(MASKBERT_+".", str(row.loc[GENERATED]), row.loc[TEMPLATE]) for _, row in templateFile.iterrows()], model_id='gpt2')['perplexities']]
    except Exception as X:
        print("EXC - getPerplexityScores: "+str(X))
        breakpoint

def getSurprisalScores(templateFile, client = None):
    print("○ Calculating surprisal score...")
    try:
        return [surpirsalWord(client, str(row.loc[GENERATED]), re.sub(MASKBERT_+".", str(row.loc[GENERATED]), row.loc[TEMPLATE])) for _, row in tqdm(templateFile.iterrows(), total=templateFile.shape[0], desc=f'Surprisal', unit=' s', position=0, leave=True)]
    except Exception as X:
        print("EXC - getSurprisalScores: "+str(X))
        breakpoint
    
#Comment the one you don't want to obtain
score_functions = {
    AFINN: getAfinnScores,
    VADER: getVaderScores,
    TEXTBLOB: getTextBlobScores,
    GOOGLE_CLOUD_NL: getGoogleCloudSentimentAnalisysScores,
    REGARD: getRegardScore,
    PERSPECTIVE: getPerspectiveScore,
    #PERPLEXITY: getPerplexityScores,
    #SURPRISAL: getSurprisalScores,
}

def evaluatePrediction(modelList):
    inputFolder, outputFolder = OUTPUT_SENTENCES, OUTPUT_EVALUATION
    os.makedirs(outputFolder, exist_ok=True)
    for modelName in modelList:
        print(f"○ Evaluating {modelName}")
        preTemplateFile, templateFile = getTemplateFile(modelName, inputFolder, outputFolder)
        if not isinstance(preTemplateFile, pd.DataFrame):
            print(f"○ {modelName} prediction file not found in the {inputFolder} folder")
            break
        templateFile = templateFile if preTemplateFile.empty else pd.concat([preTemplateFile, templateFile])
        for key, func in score_functions.items():
                client = initialize_tools[key]() if key in initialize_tools else ""
                if key == REGARD:
                    if not any(key + " "+ category in templateFile.columns for category in REGARD_CATEGORIES):
                        regScores = extractRegardScores(func(templateFile, client))
                        for category in REGARD_CATEGORIES:
                            templateFile[REGARD + " "+ category] =  regScores[category]
                    else:
                        continue
                elif key == PERSPECTIVE:
                    if not any((key + " "+ category) in templateFile.columns for category in PERSPECTIVE_CATEGORIES):
                        perspScore = extractPerspectiveScores(func(templateFile, client))
                        for category in PERSPECTIVE_CATEGORIES:
                            templateFile[PERSPECTIVE + " "+ category] =  perspScore[category]
                    else:
                        continue
                elif key not in templateFile.columns:
                    templateFile[key] = func(templateFile, client)
                templateFile.to_csv(outputFolder+modelName+'.csv', index=False)
        print(f"○ {modelName} OK!")

evaluatePrediction([DEEPSEEK])

