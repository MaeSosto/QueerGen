from lib.constants import * 
from lib.API import GOOGLE_APPLICATION_CREDENTIALS
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from afinn import Afinn
from google.cloud import language_v2
from textblob import TextBlob

def initializeGoogleCloudNL():
    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = GOOGLE_APPLICATION_CREDENTIALS
    return language_v2.LanguageServiceClient()

initialize_tools = {
    GOOGLE_CLOUD_NL: initializeGoogleCloudNL
}

def getGoogleCloudSentimentAnalisysScores(sentences, client):
    print("๏ Calculating Google Cloud Sentiment Analisys score...")
    lst = []
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
            
            lst.append(truncate(response.document_sentiment.score))
        except Exception as X:
            print("EXC - GoogleCloudSentimentAnalisysScores: "+str(X))
            lst.append(0)
    return lst
        
def getVaderScores(sentences, client = None):
    print("๏ Calculating VADER score...")
    vaderAnalyzer = SentimentIntensityAnalyzer()
    return [truncate(vaderAnalyzer.polarity_scores(sentence)['compound']) for sentence in sentences]

def getAfinnScores(sentences, client = None):
    print("๏ Calculating Afinn score...")
    afinnAnalyzer = Afinn()
    return [afinnAnalyzer.score(sentence) for sentence in sentences]

def getTextBlobScores(sentences, client = None):
    print("๏ Calculating TextBlob score...")
    return [truncate(TextBlob(sentence).sentiment[0]) for sentence in sentences]
    
def getSurprisalScores(sentences, client = None):
    print("๏ Calculating surprisal score...")
    surprisalTool = AutoHuggingFaceModel.from_pretrained('gpt2')
    surprisalList = []
    for sent in tqdm(list(sentences), position=0, leave=True):
        try:
            surp = surprisalTool.surprise(sent)[0]
            surp = surp.surprisals
            surp = truncate(s.mean(surp))
            surprisalList.append(surp)
        except Exception as X:
            print("EXC - getSurprisalScores: "+str(X))
            surprisalList.append(np.mean(surprisalList))
            continue
    return surprisalList

def getPerplexityScores(sentences, client = None):
    print("๏ Calculating perplexity score...")
    perplexityTool = load("perplexity", module_type="metric")
    perplexityList = []
    try:
        perplexityList = perplexityTool.compute(predictions=sentences, model_id='gpt2')['perplexities']
        perplexityList = [truncate(per) for per in perplexityList]
    except Exception as X:
        print("EXC - getPerplexityScoress: "+str(X))
    return perplexityList
    
score_functions = {
    GOOGLE_CLOUD_NL: getGoogleCloudSentimentAnalisysScores,
    SURPRISAL: getSurprisalScores,
    PERPLEXITY: getPerplexityScores,
    VADER: getVaderScores,
    TEXTBLOB: getTextBlobScores,
    AFINN: getAfinnScores
}

def getTemplateFile(modelName, inputFolder, outputFolder):
    print("๏ Getting the CSV file...")
    templateFile = pd.read_csv(inputFolder+modelName+".csv")
    #If the file exists already in the output folder then take that one   
    if os.path.exists(outputFolder+modelName+".csv"):
        preTemplateFile = pd.read_csv(outputFolder+modelName+".csv")
        startingFrom = preTemplateFile.shape[0]
        print(f"๏ Importing sentences from a pre-existing evaluation file [{startingFrom} sentences imported]")
        return preTemplateFile, templateFile[startingFrom:]
    else:
        print("๏ Starting from the prediction file")  
    return pd.DataFrame(), templateFile[0:]


def evaluatePredictions(modelName, inputFolder, fullSentence):
    print(f"๏ Evaluate prediction of {modelName} model...")
    outputFolder = OUTPUT_EVAL_COM if fullSentence else OUTPUT_EVAL_GEN
    preTemplateFile, templateFile = getTemplateFile(modelName, inputFolder, outputFolder)
    sentences = [(row.loc[TEMPLATE] + row.loc[GENERATED]) if fullSentence else row.loc[GENERATED] for idx, row in templateFile.iterrows()]
    
    for key, func in score_functions.items():
        client = initialize_tools[key]() if key in initialize_tools else None
        templateFile[key] = func(sentences, client)
    print("๏ Evaluation completed...")
    
    df = templateFile if preTemplateFile.empty else pd.concat([preTemplateFile, templateFile])
    os.makedirs(outputFolder, exist_ok=True)
    df.to_csv(outputFolder+modelName+'.csv', index=False)
    print("๏ File CSV generated!")

# The function `evaluatePredictions` processes model predictions, evaluates them using specified
# scoring functions, and saves the results in a CSV file.
# :param modelName: The `modelName` parameter is a string that represents the name of the model being
# evaluated. It is used to identify the specific model for which predictions are being evaluated
# :param inputFolder: The `inputFolder` parameter in the `evaluatePredictions` function refers to the folder where the input data is stored. This data is used by the model specified in `modelName` to
# generate predictions or evaluations. It could contain files, datasets, or any input data required
# for the evaluation process
# :param fullSentence: The `fullSentence` parameter in the `evaluatePredictions` function is a boolean
# flag that determines whether the evaluation should be done on full sentences or generated text only.
# If `fullSentence` is `True`, the evaluation will be performed on the combination of the template and
# generated text. Otherwise the evaluation will consider just the generated part of the sentence `
fullSentence = False
inputFolder = OUTPUT_SENTENCES_SMALL
#chosenModel = chooseModel()
MODEL_LIST = [GEMMA2, GEMMA2_27B, GEMINI_FLASH, GPT4, GPT4_MINI]
for model in MODEL_LIST:
    evaluatePredictions(model, inputFolder, fullSentence)
