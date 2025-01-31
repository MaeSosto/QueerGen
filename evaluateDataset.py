from lib.constants import * 
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from afinn import Afinn
from google.cloud import language_v2
from textblob import TextBlob

def getGoogleCloudSentimentAnalisysScores(sentences):
    print("๏ Calculating Google Cloud Sentiment Analisys score...")
    client = language_v2.LanguageServiceClient()
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
        
def getVaderScores(sentences):
    print("๏ Calculating VADER score...")
    vaderAnalyzer = SentimentIntensityAnalyzer()
    return [vaderAnalyzer.polarity_scores(sentence)['compound'] for sentence in sentences]
     
def getAfinnScores(sentences):
    print("๏ Calculating Afinn score...")
    afinnAnalyzer = Afinn()
    return [afinnAnalyzer.score(sentence) for sentence in sentences]

def getTextBlobScores(sentences):
    print("๏ Calculating TextBlob score...")
    return [ truncate(TextBlob(sentence).sentiment[0]) for sentence in sentences]
    
def getSurprisalScores(sentences):
    print("๏ Calculating surprisal score...")
    surprisalTool = AutoHuggingFaceModel.from_pretrained('gpt2')
    surprisalList = surprisalTool.surprise(list(sentences))
    return [truncate(s.mean([val for val in res.surprisals if val != np.inf])) for res in surprisalList]
        
def getPerplexityScores(sentences):
    print("๏ Calculating perplexity score...")
    perplexityTool = load("perplexity", module_type="metric")
    perplexityList = perplexityTool.compute(predictions=sentences, model_id='gpt2')['perplexities']
    return [truncate(per) for per in perplexityList]

def getTemplateFile(modelName, inputFolder, outputFolder):
    print("๏ Getting the CSV file...")
    startingFrom, preTemplateFile = 0, None
    templateFile = pd.read_csv(inputFolder+modelName+".csv")
    
    #If the file exists already in the output folder then take that one   
    if os.path.exists(outputFolder+modelName+".csv"):
        preTemplateFile = pd.read_csv(outputFolder+modelName+".csv")
        startingFrom = preTemplateFile.shape[0]
        print(f"๏ Importing sentences from a pre-existing evaluation file [{startingFrom} sentences imported]")
    else:
        print("๏ Starting from the prediction file")  
    return preTemplateFile, templateFile[startingFrom:]

def evaluatePredictions(modelName):
    fullSentence = True
    inputFolder = OUTPUT_SENTENCES_SMALL
    outputFolder = OUTPUT_EVAL_COM if fullSentence else OUTPUT_EVAL_GEN
   
    preTemplateFile, templateFile = getTemplateFile(modelName, inputFolder, outputFolder)
    
    sentences = [(row.loc[TEMPLATE] + row.loc[GENERATED]) if fullSentence else row.loc[GENERATED] for idx, row in templateFile.iterrows()]
    templateFile[VADER] = getVaderScores(sentences)
    templateFile[TEXTBLOB] = getTextBlobScores(sentences)
    templateFile[AFINN] = getAfinnScores(sentences)
    templateFile[GOOGLE_CLOUD_NL] = getGoogleCloudSentimentAnalisysScores(sentences)
    templateFile[PERPLEXITY] = getPerplexityScores(sentences)
    templateFile[SURPRISAL] = getSurprisalScores(sentences)
    print("๏ Evaluation completed...")
    
    df = templateFile if preTemplateFile.empty else pd.concat([preTemplateFile, templateFile])
    os.makedirs(outputFolder, exist_ok=True)
    df.to_csv(outputFolder+modelName+'.csv', index=False)
    print("๏ File CSV generated!")
    
chosenModel = chooseModel()
evaluatePredictions(MODEL_LIST[chosenModel])