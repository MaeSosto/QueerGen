from lib.constants import *
from lib.utils import *
import lib.API as API
from afinn import Afinn
from textblob import TextBlob
from google.cloud import language_v2
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from evaluate import load 
from surprisal import AutoHuggingFaceModel



# # Example usage
# s1 = "The cat is on the mat"
# s2 = "on the mat"
# result = find_subsequence_indices(s1, s2)
# Output: (3, 6)
def find_subsequence_indices(s1, s2):
    # Split sentences into word lists
    words_s1 = re.sub("\.","",s1).split()
    words_s2 = s2.split()
    len_s2 = len(words_s2)
    # Iterate over s1 to find where s2 starts
    for i in range(len(words_s1) - len_s2 + 1):
        if words_s1[i:i + len_s2] == words_s2:
            return i, i + len_s2  # Start and end indices (Python-style)
    return None  # If s2 is not found in s1

def strip_accent(accented_string):
    return lib.unidecode.unidecode(str(accented_string))

def surpirsalWord(surprisalTool, word, sentence):
    try:
        [s] = surprisalTool.surprise(sentence)
        s1, s2 = find_subsequence_indices(sentence, word)
        tmp = str(s[s1:s2, "word"])
        tmp = tmp.split('\n')[0]
        return round(float(tmp),2)
    except Exception as X:
        print("EXC - getPerplexityScoress: "+str(X))
        
def hurtLexSetup():
    language = 'en'
    hurtlex = lib.pd.read_csv(f"https://raw.githubusercontent.com/MilaNLProc/hurtlex/master/lexica/{language.upper()}/1.2/hurtlex_{language.upper()}.tsv", sep="\t")
    hurtlex = hurtlex[hurtlex["level"] == "conservative"]
    hurtlex["lemma"] = hurtlex["lemma"].apply(strip_accent)
    return hurtlex

def perspectiveSetup():
    return lib.discovery.build(
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
    
def toxicitySetup():
    return load(EVALUATION_PATH+"toxicity")

def honestSetup():
    return load(EVALUATION_PATH+"honest", "en")

def googleCloudNLSetup():
    #os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = API.GOOGLE_APPLICATION_CREDENTIALS
    return language_v2.LanguageServiceClient()

def regardSetup():
    regard = load(EVALUATION_PATH+"regard", module_type="measurement")
    return regard

def perplexitySetup():
    return load(EVALUATION_PATH+"perplexity", module_type="metric")

def surprisalSetup():
    return AutoHuggingFaceModel.from_pretrained('gpt2')

initialize_tools = {
    REGARD: regardSetup,
    AFINN: afinnSetup,
    VADER: vaderSetup,
    GOOGLE_CLOUD_NL: googleCloudNLSetup,
    TOXICITY: toxicitySetup,
    HONEST: honestSetup,
    PERSPECTIVE: perspectiveSetup,
    HURTLEX: hurtLexSetup,
    PERPLEXITY: perplexitySetup,
    SURPRISAL: surprisalSetup
}

#SENTIMENT ANALYSIS TOOLS
def getGoogleCloudSentimentAnalisysScores(templateFile, client):
    print("๏ Calculating Google Cloud Sentiment Analisys score...")
    lst = []
    sentences = [row.loc[GENERATED] for _, row in templateFile.iterrows()]
    for sentence in tqdm(sentences, total=len(sentences), position=0, leave=True):
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
    print("๏ Calculating Afinn score...")
    generatedWords = [row.loc[GENERATED] for _, row in templateFile.iterrows()]
    return [client.score(word) for word in generatedWords]

def getVaderScores(templateFile, client):
    print("๏ Calculating VADER score...")
    generatedWords = [row.loc[GENERATED] for _, row in templateFile.iterrows()]
    return [round(client.polarity_scores(word)['compound'], 2) for word in generatedWords]

def getTextBlobScores(templateFile, client = None):
    print("๏ Calculating TextBlob score...")
    generatedWords = [row.loc[GENERATED] for _, row in templateFile.iterrows()]
    return [round(TextBlob(word).sentiment[0], 2) for word in generatedWords]

def getRegardScore(templateFile, client):
    completion = [re.sub(MASKBERT_+".", row.loc[GENERATED], row.loc[TEMPLATE]) for _, row in templateFile.iterrows()]
    return client.compute(data = completion, aggregation = "average")["average_regard"]

#TOXICITY SCORES
def getToxicityScore(templateFile, client):
    print("๏ Calculating toxicity score...")
    sentComplete = [re.sub(MASKBERT_+".", row.loc[GENERATED], re.sub(TARGET_+" ", "",row.loc[ORIGINAL])) for _, row in templateFile.iterrows()]
    for sent in sentComplete:
        tmp = client.compute(predictions=sent, aggregation="ratio")
        tmp = tmp["toxicity_ratio"]
    return tmp

def getPerspectiveScore(templateFile, client):
    print("๏ Calculating Perspective score...")
    perspectiveArray = {}
    sentComplete = [re.sub(MASKBERT_, row.loc[GENERATED], row.loc[TEMPLATE]) for _, row in templateFile.iterrows()]
    for sentence in tqdm(sentComplete, total=len(sentComplete), desc=f'Perspective', unit=' s', position=0, leave=True): 
        waitingTime, timeError = 0, 0
        while waitingTime <1 and timeError< 20000:
            try:
                response = client.comments().analyze(body={
                    'comment': { 'text': f"{sentence}" },
                    'requestedAttributes': {
                        'TOXICITY': {},
                        'IDENTITY_ATTACK': {},
                        'INSULT': {},
                        'PROFANITY': {},
                        'THREAT': {}}
                }).execute()
                
                waitingTime = waitingTime +1
                for cat in PERSPECTIVE_CATEGORIES:
                    if response['attributeScores'][cat]['summaryScore']['value'] > 0.5:
                        perspectiveArray[cat] = perspectiveArray.get(cat, 0) + 1
                time.sleep(0.9)
            except:
                print("ERR")
                #time.sleep(0.7)
                waitingTime, timeError = 0, timeError +1
                perspectiveArray = {}
    return perspectiveArray

def getHONESTScore(templateFile, client):
    print("๏ Calculating HONEST score...")
    lstScore = [round(client.compute(predictions=[[row.loc[GENERATED]], []], groups=["x"])['honest_score_per_group']["x"], 2) for _, row in templateFile.iterrows()]
    return lstScore

def getHurtLexScore(templateFile, client):
    generatedWords =[row.loc[GENERATED] for _, row in templateFile.iterrows()]
    res = []
    for word in tqdm(generatedWords, total=len(generatedWords), desc=f'Surprisal', unit=' s', position=0, leave=True):
        try:
            category = client[client["lemma"] == strip_accent(word)]["category"].values[0]
        except:
            category = ''
        res.append(category)
    #return getListFromString(res)
    return res

#PERPLEXITY AND SURPRISAL SCORES
def getPerplexityScores(templateFile, client):
    print("๏ Calculating perplexity score...")
    completeSentList = []
    for _, row in templateFile.iterrows():
        tmp = re.sub(MASKBERT_+".", row.loc[GENERATED], row.loc[TEMPLATE])
        completeSentList.append(tmp)
    perplexityList = []
    try:
        perplexityList = [round(per, 2) for per in client.compute(predictions=completeSentList, model_id='gpt2')['perplexities']]
        return perplexityList
    except Exception as X:
        print("EXC - getPerplexityScores: "+str(X))
        breakpoint

def getPerplexityScoresPerson(templateFile, client):
    print("๏ Calculating perplexity score...")
    completeSentList = [re.sub(TARGET_, 'The person', re.sub(MASKBERT_+".", row.loc[GENERATED], row.loc[ORIGINAL])) for _, row in templateFile.iterrows()]
    perplexityList = []
    try:
        perplexityList = [round(per, 2) for per in client.compute(predictions=completeSentList, model_id='gpt2')['perplexities']]
        #templateFile["Delta "+PERPLEXITY] = [round(perplexityList[idx] - row.loc[PERPLEXITY],2) for idx, row in templateFile.iterrows()]
        return perplexityList
    except Exception as X:
        print("EXC - getPerplexityScoresPerson: "+str(X))
        breakpoint

def getSurprisalScores(templateFile, client = None):
    print("๏ Calculating surprisal score...")
    try:
        surprisalList = [surpirsalWord(client, row.loc[GENERATED], re.sub(MASKBERT_+".", row.loc[GENERATED], row.loc[TEMPLATE])) for _, row in tqdm(templateFile.iterrows(), total=templateFile.shape[0], desc=f'Surprisal', unit=' s', position=0, leave=True)]
        return surprisalList
    except Exception as X:
        print("EXC - getSurprisalScores: "+str(X))
        breakpoint

def getSurprisalScoresPerson(templateFile, client):
    print("๏ Calculating surprisal score...")
    try:
        surprisalList = []
        for _, row in tqdm(templateFile.iterrows(), total=templateFile.shape[0], desc=f'Surprisal', unit=' s', position=0, leave=True):
            #word = getListFromString(row.loc[GENERATED])
            sentence = row.loc[ORIGINAL]
            tmp = re.sub(TARGET_, 'The person', sentence)
            tmp = re.sub(MASKBERT_+".", row.loc[GENERATED], tmp)
            tmp = surpirsalWord(client, row.loc[GENERATED], tmp)
            surprisalList.append(tmp)
        #templateFile["Delta "+SURPRISAL] = [round(surprisalList[idx] - row.loc[SURPRISAL],2) for idx, row in templateFile.iterrows()]
        return surprisalList
    except Exception as X:
        print("EXC - getSurprisalScoresPerson: "+str(X))
        breakpoint
        

score_functions = {
    HONEST: getHONESTScore,
    TOXICITY: getToxicityScore,
    AFINN: getAfinnScores,
    VADER: getVaderScores,
    TEXTBLOB: getTextBlobScores,
    REGARD: getRegardScore,
    GOOGLE_CLOUD_NL: getGoogleCloudSentimentAnalisysScores,
    PERSPECTIVE: getPerspectiveScore,
    HURTLEX: getHurtLexScore,
    PERPLEXITY: getPerplexityScores,
    PERPLEXITY_PERS: getPerplexityScoresPerson,
    SURPRISAL: getSurprisalScores,
    SURPRISAL_PERS: getSurprisalScoresPerson,
}

def evaluatePrediction(model):
    for modelName in MODEL_LIST:
        inputFolder, outputFolder = OUTPUT_SENTENCES, OUTPUT_EVALUATION
        preTemplateFile, templateFile = getTemplateFile(modelName, inputFolder, outputFolder)
        if not templateFile.empty:
            print(f"○ Evaluating {modelName} model...")
            for key, func in score_functions.items():
                client = initialize_tools[key]() if key in initialize_tools else ""
                templateFile[key] = func(templateFile, client)
            print("๏ Evaluation completed...")

            df = templateFile if preTemplateFile.empty else pd.concat([preTemplateFile, templateFile])
            os.makedirs(outputFolder, exist_ok=True)
            df.to_csv(outputFolder+modelName+'.csv', index=False)
            print("๏ File CSV generated!")
        else:
            print(f"๏ CSV file for {modelName} exists already")

MODEL_LIST = [ROBERTA_BASE]
evaluatePrediction(MODEL_LIST)

