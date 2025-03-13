from lib.constants import * 
import math
from evaluate import load 

def getScores(fileName, folderPath):
    templateFile = pd.read_csv(folderPath+ fileName)
    templateFile.replace(pd.NA, 0)
    modelName = fileName.replace('.csv', '')
    print(f"๏ Getting scores from {fileName}...")
    
    #Create scoreCollection
    scoreCollection = {}
    for subj in NOUN_TYPES:
        scoreCollection[modelName + " "+ subj] = {}
        for evalTool in EVALUATION_TOOLS:
            if evalTool == PERSPECTIVE:
                for cat in PERSPECTIVE_CATEGORIES:
                    scoreCollection[modelName + " "+ subj][evalTool+ " "+ cat] = []
            if evalTool == REGARD:
                for cat in REGARD_CATEGORIES:
                    scoreCollection[modelName + " "+ subj][evalTool+ " "+ cat] = []
            else:
                scoreCollection[modelName + " "+ subj][evalTool] = []
    #Fill scoreCollection      
    for _, row in templateFile.iterrows():
        for evalTool in EVALUATION_TOOLS:
            key = modelName + " "+ row.loc[TYPE]
            if evalTool == REGARD:
                for cat in REGARD_CATEGORIES:
                    scoreCollection[key][evalTool+ " " + cat].append(row.loc[evalTool+ " " + cat])
            elif evalTool == HURTLEX and isinstance(val, str):
                scoreCollection[key][evalTool].append(row.loc[evalTool])
            elif evalTool == PERSPECTIVE:
               for cat in PERSPECTIVE_CATEGORIES:
                    val = row.loc[evalTool+ " " + cat]
                    scoreCollection[key][evalTool+ " " + cat].append(val)
                #scoreCollection[key][tool].append(len(val))
            else:
                val = row.loc[evalTool]
                scoreCollection[key][evalTool].append(val)

    #Calculate overall scores
    for subj in NOUN_TYPES:
        scoresArray = [] 
        columnHeader = []
        for evalTool in EVALUATION_TOOLS:
            if evalTool == REGARD:
                for cat in REGARD_CATEGORIES:
                    columnHeader.append(evalTool+ " "+ cat)
            elif evalTool == PERSPECTIVE:
                for cat in PERSPECTIVE_CATEGORIES:
                    columnHeader.append(evalTool+ " "+ cat)
            else:
                columnHeader.append(evalTool)
        for evalTool in EVALUATION_TOOLS:
            if evalTool == HURTLEX:
                scoresArray.append(len(scoreCollection[modelName + " "+ subj][evalTool]))    
            elif evalTool == REGARD:
                score = [np.mean(scoreCollection[modelName + " "+ subj][evalTool+ " " + cat]) for cat in REGARD_CATEGORIES] 
                #score = np.mean(score)
                [scoresArray.append(round(s,2)) for s in score] 
            elif evalTool == PERSPECTIVE:
                score = [sum(scoreCollection[modelName + " "+ subj][evalTool+ " " + cat]) for cat in PERSPECTIVE_CATEGORIES]
                #score = sum(score)
                [scoresArray.append(round(s,2)) for s in score] 
            elif evalTool in [ev for ev in EVALUATION_TOOLS if ev not in [HURTLEX, PERSPECTIVE, REGARD]]:
                scoresArray.append(round(np.mean(scoreCollection[modelName + " "+ subj][evalTool]), 2))
        scoreCollection[modelName + " "+ subj] = scoresArray
    #print(scoreCollection)

    dfScore = pd.DataFrame.from_dict(scoreCollection, orient='index', columns=columnHeader)
    return dfScore  

def overallScores(folderPath):
    dfOverall = pd.DataFrame()
    for file in [f for f in os.listdir(folderPath) if f != 'results.csv']:
        dfModel = getScores(file, folderPath)
        dfOverall = dfModel if dfOverall.empty else pd.concat([dfOverall, dfModel])
        dfOverall.to_csv(folderPath + 'results.csv')
    print("๏ File resutls.csv generated!")

# The function `overallScores` reads scores from files in a specified folder, combines them into a
# DataFrame, and saves the results to a CSV file.
# :param folderPath: The `folderPath` parameter in the `overallScores` function is a string that
# represents the path to a directory where files are stored. The function reads files from this
# directory, processes them, and generates an output file named `results.csv` in the same directory
folderPath = OUTPUT_EVALUATION
#overallScores(folderPath)
dfModel = getScores('BERT_large.csv', folderPath)