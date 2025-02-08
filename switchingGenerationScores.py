from lib.constants import * 
from lib.utils import find_subsequence_indices
from evaluate import load 
from surprisal import AutoHuggingFaceModel
import numpy as np
import statistics as s

surprisalTool = AutoHuggingFaceModel.from_pretrained('gpt2')

scores = {
    QUEER: {
        NON_QUEER: [],
        NEUTRAL: []
    },
    NON_QUEER: {
        QUEER: [],
        NEUTRAL: []
    },
    NEUTRAL:{
        QUEER: [],
        NON_QUEER: [],
    }
}

score_collection = defaultdict(lambda: defaultdict(list))

def getScores(modelName):
    print("๏ Getting the CSV file...")
    generatedFile = pd.read_csv(OUTPUT_SENTENCES+ modelName+'.csv')
        
    for _, currentRow in tqdm(generatedFile.iterrows(), position=0, leave=True):
        currentType = currentRow.loc[TYPE]
        currentSentence = currentRow.loc[TEMPLATE] + currentRow.loc[GENERATED]
        [s] = surprisalTool.surprise(currentSentence)
        s1, s2 = find_subsequence_indices(currentSentence, currentRow.loc[GENERATED])
        tmp = str(s[s1:s2, "word"])
        currentSurpisal = float(tmp.split('\n')[0])
        for _, otherRow in generatedFile.iterrows():
            otherType = otherRow.loc[TYPE]
            if currentType != otherType:
                otherSentence = currentRow.loc[TEMPLATE] + otherRow.loc[GENERATED]
                [s] = surprisalTool.surprise(otherSentence)
                s1, s2 = find_subsequence_indices(otherSentence, otherRow.loc[GENERATED])
                otherSurprisal = float(str(s[s1:s2, "word"]).split('\n')[0])
                score_collection[currentRow.loc[TYPE]][otherRow.loc[TYPE]].append(currentSurpisal - otherSurprisal)
    
    for s in score_collection.iterrows():
        for c in s.iterrows():
            score_collection[s][c] = truncate(np.mean(score_collection[s][c]))
            
    print(score_collection) 
        
        
        # for _, currentRow in generatedFile.iterrows():
        
        
    # sentencesPerson = []
    # for idx, row in templateFile.iterrows():
    #     originalTemp = str(row.loc['original '+TEMPLATE])
    #     init = True if originalTemp.split(TARGET_)[0] == '' else False
    #     the = 'The ' if init else 'the '
    #     generatedPart = generatedFile.loc[idx][GENERATED]
    #     sentencesPerson.append(originalTemp.replace(TARGET_, the + "person") + generatedPart)
    
    # generatedFile[PERPLEXITY] = getPerplexityScores(sentences)
    # generatedFile[SURPRISAL] = getSurprisalScores(sentences)
    # generatedFile[PERPLEXITY + " person"] = getPerplexityScores(sentencesPerson)
    # generatedFile[SURPRISAL + " person"] = getSurprisalScores(sentencesPerson)

    # #df = pd.DataFrame.from_dict(templateFile)    
    # generatedFile.to_csv(OUTPUT_EVAL_COM+ modelName+'.csv', index=False)
    # print("๏ File updated correctly!")

getScores(LLAMA3)