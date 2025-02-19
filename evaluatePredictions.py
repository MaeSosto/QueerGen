from lib.constants import *
from lib.utils import *
from afinn import Afinn
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

def getVaderScores(sentences):
    print("๏ Calculating VADER score...")
    vaderAnalyzer = SentimentIntensityAnalyzer()
    return [round(vaderAnalyzer.polarity_scores(sentence)['compound'], 2) for sentence in sentences]

def getAfinnScores(sentences):
    print("๏ Calculating Afinn score...")
    afinnAnalyzer = Afinn()
    return [afinnAnalyzer.score(sentence) for sentence in sentences]

def getTextBlobScores(sentences):
    print("๏ Calculating TextBlob score...")
    return [round(TextBlob(sentence).sentiment[0], 2) for sentence in sentences]

def getPerplexityScores(sentences):
    print("๏ Calculating perplexity score...")
    perplexityTool = load("perplexity", module_type="metric")
    perplexityList = []
    try:
        perplexityList = perplexityTool.compute(predictions=sentences, model_id='gpt2')['perplexities']
        perplexityList = [round(per, 2) for per in perplexityList]
    except Exception as X:
        print("EXC - getPerplexityScoress: "+str(X))
    return perplexityList

def getSurprisalScores(sentences):
    print("๏ Calculating surprisal score...")
    surprisalTool = AutoHuggingFaceModel.from_pretrained('gpt2')
    surprisalList = []
    for sent in tqdm(list(sentences), position=0, leave=True):
        try:
            surp = surprisalTool.surprise(sent)[0]
            surp = surp.surprisals
            surp = round(s.mean(surp), 2)
            surprisalList.append(surp)
        except Exception as X:
            print("EXC - getSurprisalScores: "+str(X))
            surprisalList.append(np.mean(surprisalList))
            continue
    return surprisalList

score_functions = {
    VADER: getVaderScores,
    TEXTBLOB: getTextBlobScores,
    AFINN: getAfinnScores,
    PERPLEXITY: getPerplexityScores,
    SURPRISAL: getSurprisalScores
}




def getTemplateFile(modelName, inputFolder, outputFolder, predictionConsidered =1):
    print("๏ Getting the CSV file...")
    templateFile = pd.read_csv(f"{inputFolder}{modelName}_{predictionConsidered}.csv")
    #If the file exists already in the output folder then take that one   
    if os.path.exists(outputFolder+modelName+".csv"):
        preTemplateFile = pd.read_csv(outputFolder+modelName+".csv")
        startingFrom = preTemplateFile.shape[0]
        print(f"๏ Importing sentences from a pre-existing evaluation file [{startingFrom} sentences imported]")
        return preTemplateFile, templateFile[startingFrom:]
    else:
        print("๏ Starting from the prediction file")  
    return pd.DataFrame(), templateFile[0:]


def evaluatePrediction(modelName, predictionsConsidered = 1):
    print("○ Evaluatior running...")
    inputFolder, outputFolder = OUTPUT_SENTENCES, OUTPUT_EVALUATION
    preTemplateFile, templateFile = getTemplateFile(modelName, inputFolder, outputFolder, predictionsConsidered)
    os.makedirs(outputFolder, exist_ok=True)
    
    predictedWords = [row.loc[GENERATED] for id_, row in templateFile.iterrows()]
        
    for key, func in score_functions.items():
        templateFile[key] = func(predictedWords)
    print("๏ Evaluation completed...")

    df = templateFile if preTemplateFile.empty else pd.concat([preTemplateFile, templateFile])
    os.makedirs(outputFolder, exist_ok=True)
    df.to_csv(outputFolder+modelName+'.csv', index=False)
    print("๏ File CSV generated!")

predictionsConsidered = 1
MODEL_LIST = [BERT_BASE, LLAMA3, GPT4]
for model in MODEL_LIST:
    evaluatePrediction(model)

