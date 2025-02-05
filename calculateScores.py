from lib.constants import * 
from evaluate import load 
from surprisal import AutoHuggingFaceModel
import numpy as np
import statistics as s

def getSurprisalScores(sentences):
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

def getPerplexityScores(sentences):
    print("๏ Calculating perplexity score...")
    perplexityTool = load("perplexity", module_type="metric")
    perplexityList = []
    try:
        perplexityList = perplexityTool.compute(predictions=sentences, model_id='gpt2')['perplexities']
        perplexityList = [truncate(per) for per in perplexityList]
    except Exception as X:
        print("EXC - getPerplexityScoress: "+str(X))
    return perplexityList


def getScores(modelName):
    print("๏ Getting the CSV file...")
    generatedFile = pd.read_csv(OUTPUT_SENTENCES+ modelName+'.csv')
    templateFile = pd.read_csv(TEMPLATES_COMPLETE_PATH)
    sentences = [(row.loc[TEMPLATE] + row.loc[GENERATED]) for idx, row in generatedFile.iterrows()]
    
    sentencesPerson = []
    for idx, row in templateFile.iterrows():
        originalTemp = str(row.loc['original '+TEMPLATE])
        init = True if originalTemp.split(TARGET_)[0] == '' else False
        the = 'The ' if init else 'the '
        generatedPart = generatedFile.loc[idx][GENERATED]
        sentencesPerson.append(originalTemp.replace(TARGET_, the + "person") + generatedPart)
    
    generatedFile[PERPLEXITY] = getPerplexityScores(sentences)
    generatedFile[SURPRISAL] = getSurprisalScores(sentences)
    generatedFile[PERPLEXITY + " person"] = getPerplexityScores(sentencesPerson)
    generatedFile[SURPRISAL + " person"] = getSurprisalScores(sentencesPerson)

    #df = pd.DataFrame.from_dict(templateFile)    
    generatedFile.to_csv(OUTPUT_EVAL_COM+ modelName+'.csv', index=False)
    print("๏ File updated correctly!")

getScores('llama3')