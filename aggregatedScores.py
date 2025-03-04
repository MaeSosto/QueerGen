from lib.constants import * 
import math
from evaluate import load 

def getScores(fileName, folderPath):
    templateFile = pd.read_csv(folderPath+ fileName)
    modelName = fileName.replace('.csv', '')
    print(f"๏ Getting scores from {fileName}...")
    
    #Create scoreCollection
    scoreCollection = {}
    for subj in NOUN_TYPES:
        scoreCollection[modelName + " "+ subj] = {}
        for tool in EVALUATION_TOOLS:
            if tool == PERSPECTIVE:
                for cat in PERSPECTIVE_CATEGORIES:
                    scoreCollection[modelName + " "+ subj][tool+ " "+ cat] = []
            if tool == REGARD:
                for cat in REGARD_CATEGORIES:
                    scoreCollection[modelName + " "+ subj][tool+ " "+ cat] = []
            else:
                scoreCollection[modelName + " "+ subj][tool] = []
    #Fill scoreCollection      
    for _, row in templateFile.iterrows():
        for tool in EVALUATION_TOOLS:
            type = row.loc[TYPE]
            if tool == REGARD:
                for cat in REGARD_CATEGORIES:
                    scoreCollection[modelName + " "+ type][tool+ " " + cat].append(row.loc[tool+ " " + cat])
            elif tool == HURTLEX and isinstance(val, str):
                scoreCollection[modelName + " "+ type][tool].append(row.loc[tool])
            elif tool == PERSPECTIVE:
               for cat in PERSPECTIVE_CATEGORIES:
                    val = row.loc[tool+ " " + cat]
                    scoreCollection[modelName + " "+ type][tool+ " " + cat].append(val)
                #scoreCollection[modelName + " "+ type][tool].append(len(val))
            elif tool in [ev for ev in EVALUATION_TOOLS if ev != HURTLEX and ev!= PERSPECTIVE and ev!= REGARD ]:
                scoreCollection[modelName + " "+ type][tool].append(row.loc[tool])

    #Calculate overall scores
    for subj in NOUN_TYPES:
        scoresArray = [] 
        columnHeader = []
        for tool in EVALUATION_TOOLS:
            if tool == REGARD:
                for cat in REGARD_CATEGORIES:
                    columnHeader.append(tool+ " "+ cat)
            elif tool == PERSPECTIVE:
                for cat in PERSPECTIVE_CATEGORIES:
                    columnHeader.append(tool+ " "+ cat)
            else:
                columnHeader.append(tool)
        for tool in EVALUATION_TOOLS:
            if tool == HURTLEX:
                scoresArray.append(len(scoreCollection[modelName + " "+ subj][tool]))    
            elif tool == REGARD:
                score = [np.mean(scoreCollection[modelName + " "+ subj][tool+ " " + cat]) for cat in REGARD_CATEGORIES] 
                #score = np.mean(score)
                [scoresArray.append(round(s,2)) for s in score] 
            elif tool == PERSPECTIVE:
                score = [sum(scoreCollection[modelName + " "+ subj][tool+ " " + cat]) for cat in PERSPECTIVE_CATEGORIES]
                #score = sum(score)
                [scoresArray.append(round(s,2)) for s in score] 
            elif tool in [ev for ev in EVALUATION_TOOLS if ev not in [HURTLEX, PERSPECTIVE, REGARD]]:
                scoresArray.append(round(np.mean(scoreCollection[modelName + " "+ subj][tool]), 2))
        scoreCollection[modelName + " "+ subj] = scoresArray
    print(scoreCollection)

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
overallScores(folderPath)