from constants import * 
from LIB_evaluation import *

OUTPUT_EVAL_COM = 'evaluation_completeSentence/'
OUTPUT_EVAL_GEN = 'evaluation_generatedPart/'

def newDicSentences():
    return {
        TYPE: [],
        TEMPLATE: [],
        GENERATED: [],
        VADER: [],
        #PERSPECTIVE: [],
        GOOGLE_CLOUD_NL: [],
        TEXTBLOB: [],
        AFINN: []
    }


def newScoresDict(modelName):
    return {
        modelName + " "+ QUEER : {
            VADER: [],
            #PERSPECTIVE: [],
            GOOGLE_CLOUD_NL: [],
            TEXTBLOB: [],
            AFINN: []
        },
        modelName + " "+ NON_QUEER : {
            VADER: [],
            #PERSPECTIVE: [],
            GOOGLE_CLOUD_NL: [],
            TEXTBLOB: [],
            AFINN: []
        },
        modelName + " "+ NEUTRAL : {
            VADER: [],
            #PERSPECTIVE: [],
            GOOGLE_CLOUD_NL: [],
            TEXTBLOB: [],
            AFINN: []
        }  
    }
   

def getListFromString(text):
    text = re.sub(r"'", "", str(text))
    text = re.sub(r'\]', '', text)
    text = re.sub(r'\[', '', text)
    return list(map(int, text.split(",")))

def calcAverageScores(val):
    #val[PERSPECTIVE] = [sum(getListFromString(per)) for per in val[PERSPECTIVE]]
    scores = []
    for met in EVALUATION_TOOLS:
        scores.append(truncate(sum(val[met])/len(val[met])))
    return scores
    
def truncate(float_number, decimal_places = 2):
    multiplier = 10 ** decimal_places
    return int(float_number * multiplier) / multiplier if float_number != None else 0

# def perspectiveSetup():
#     client = discovery.build(
#         "commentanalyzer",
#         "v1alpha1",
#         developerKey=API.PERSPECTIVE_API_KEY,
#         discoveryServiceUrl="https://commentanalyzer.googleapis.com/$discovery/rest?version=v1alpha1",
#         static_discovery=False,
#     )
#     return client


# def prespectiveEvaluator(sentence, prespectiveAPI):
#     perspectiveArray = [0, 0, 0, 0, 0]
#     waitingTime, timeError = 0, 0
#     while waitingTime <1 and timeError< 20:
#         try:
#             response = prespectiveAPI.comments().analyze(body={
#                 'comment': { 'text': f"{sentence}" },
#                 'requestedAttributes': {
#                     'TOXICITY': {},
#                     'IDENTITY_ATTACK': {},
#                     'INSULT': {},
#                     'PROFANITY': {},
#                     'THREAT': {}}
#             }).execute()
            
#             waitingTime = waitingTime +1
#             for idx, cat in enumerate(PERSPECTIVE_CATEGORIES):
#                 if response['attributeScores'][cat]['summaryScore']['value'] > 0.5:
#                     perspectiveArray[idx] = 1
#             time.sleep(0.9)
#         except Exception as X:
#             print("EXC: "+str(X))
#             time.sleep(0.7)
#             waitingTime, timeError = 0, timeError +1
#             #perspectiveArray = []
#         return perspectiveArray


def NLPCloudSentimentAnalysis(text_content):
    client = language_v2.LanguageServiceClient()
    try:
        response = client.analyze_sentiment(request={
            "document": {
                "content": text_content,
                "type_": language_v2.Document.Type.PLAIN_TEXT,
                "language_code": "en"
            }, 
            "encoding_type": language_v2.EncodingType.UTF8
        })
        
        return response.document_sentiment.score
    except Exception as X:
        print("EXC: "+str(X))
        return None

def getResultsScores(modelName, templateFile, outputPath):
    scores = newScoresDict(modelName)
    
    newDf = pd.DataFrame(columns=EVALUATION_TOOLS,)
    for index,row in templateFile.iterrows():
        for eval in EVALUATION_TOOLS:
            scores[modelName + " "+row[TYPE]][eval].append(row.loc[eval])
    for sub in SUBJECT_TYPES:
        newDf.loc[modelName + " "+sub] = calcAverageScores(scores[modelName + " "+sub])
    os.makedirs(outputPath, exist_ok=True)
    if os.path.exists(outputPath+ 'results.csv'):
        print("๏ Getting previous results file")
        try:
            previousDf = pd.read_csv(outputPath+ 'results.csv', index_col='index')
            data_total = pd.concat([previousDf, newDf])
            data_total.to_csv(outputPath+ 'results.csv', index_label = 'index')
            print('๏ Results file updated!')  
        except Exception as e:
            print(e)
            print('๏ Error in updating file!') 
    else:
        print("๏ Creating a new results file")
        try:
            data_total = pd.DataFrame.from_dict(newDf)
            
            data_total.to_csv(outputPath+ 'results.csv', index_label = 'index')
            print('๏ Results file generated!')  
        except Exception as e:
            print(e)
            print('๏ Error in updating file!') 
    
        
      
def preExistingFile(outputPath, outputFilePath):
    startingFrom = 0
    dicSentences = newDicSentences()  
    
    os.makedirs(outputPath, exist_ok=True)
    #If the file exists already in the output folder then take that one   
    if os.path.exists(outputFilePath):
        df = pd.read_csv(outputFilePath, index_col=None)
        startingFrom = df.shape[0]
        print(f"๏ Importing sentences from a pre-existing evaluation file [{startingFrom} sentences imported]")
        for idx, row in df.iterrows():
            dicSentences[TYPE].append(row.loc[TYPE])
            dicSentences[TEMPLATE].append(row.loc[TEMPLATE])
            dicSentences[GENERATED].append(row.loc[GENERATED])
            for tool in EVALUATION_TOOLS:
                dicSentences[tool].append(row.loc[tool])
        #df = pd.DataFrame.from_dict(dicSentences)    
        print("๏ Sentences imported correctly!")
    else:
        print("๏ Starting from the prediction file")  
    return startingFrom, dicSentences
      
def evaluatePredictions(modelName, fullSentence, inputPath, outputPath = OUTPUT_EVALUATION):
    analyzer = SentimentIntensityAnalyzer()
    #prespectiveAPI = perspectiveSetup()
    global afinn
    afinn = Afinn()
    
    inputFilePath = inputPath+modelName+".csv"
    outputFilePath = outputPath+modelName+'.csv'

    #dicSentences = newDicSentences()

    #Checking if there is an existing file with evaluations
    startingFrom, dicSentences = preExistingFile(outputPath, outputFilePath)
    
    templateFile = pd.read_csv(inputFilePath)
    templateFile = templateFile[startingFrom:]
    df = pd.DataFrame.from_dict(dicSentences)    
    for index,row in tqdm(templateFile.iterrows(), total=templateFile.shape[0], desc=f'Evaluating {modelName}\'s answers', unit=' sentences'):
        if fullSentence:
            sentence = row.loc[GENERATED]
        else:
            sentence = re.sub(row.loc[TEMPLATE], "", row.loc[GENERATED])
        dicSentences[TYPE].append(row.loc[TYPE])
        dicSentences[TEMPLATE].append(row.loc[TEMPLATE])
        dicSentences[GENERATED].append(sentence)
        dicSentences[VADER].append(analyzer.polarity_scores(sentence)['compound'])
        #dicSentences[PERSPECTIVE].append(prespectiveEvaluator(sentence, prespectiveAPI))
        dicSentences[GOOGLE_CLOUD_NL].append(truncate(NLPCloudSentimentAnalysis(sentence)))
        dicSentences[TEXTBLOB].append(truncate(TextBlob(sentence).sentiment[0]))
        dicSentences[AFINN].append(afinn.score(sentence))
        df = pd.DataFrame.from_dict(dicSentences)    
        os.makedirs(outputPath, exist_ok=True)
        df.to_csv(outputFilePath, index_label = 'index')
    print('๏ Get scores ')
    getResultsScores(modelName, df, outputPath)

# chosenModel = -1
# while chosenModel < 0 or chosenModel > len(MODEL_LIST)-1:
#     print('๏ Select a model: ')
#     for idx, x in enumerate(MODEL_LIST):
#         print(f"[{idx}] -  {x}")
#     chosenModel = int(input())

# evaluatePredictions(MODEL_LIST[chosenModel])

#Model list to evaluate
models = [LLAMA3_70B, GPT4, GEMMA2]
for m in models:
    #Input: model, True if considering the whole sentence False considering only the generated part, input path, output path
    evaluatePredictions(m, True, OUTPUT_PREDICTION, OUTPUT_EVAL_COM)