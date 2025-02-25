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
            scoreCollection[modelName + " "+ subj][tool] = []
    #Fill scoreCollection      
    for idx, row in templateFile.iterrows():
        for tool in EVALUATION_TOOLS:
            type = row.loc[TYPE]
            val = row.loc[tool]
            if tool == HURTLEX and isinstance(val, str):
                scoreCollection[modelName + " "+ type][tool].append(val)
            elif tool == PERSPECTIVE and isinstance(val, dict):
                scoreCollection[modelName + " "+ type][tool].append(len(val))
            elif tool in [ev for ev in EVALUATION_TOOLS if ev != HURTLEX and ev!= PERSPECTIVE ]:
                scoreCollection[modelName + " "+ type][tool].append(val)

    #Calculate overall scores
    for subj in NOUN_TYPES:
        scoresArray = []    
        for tool in EVALUATION_TOOLS + [TOXICITY, HONEST, REGARD]:
            if tool == HURTLEX or tool == PERSPECTIVE:
                scoresArray.append(len(scoreCollection[modelName + " "+ subj][tool]))    
            elif tool in [ev for ev in EVALUATION_TOOLS if ev not in [HURTLEX, PERSPECTIVE]]:
                scoresArray.append(round(np.mean(scoreCollection[modelName + " "+ subj][tool]), 2))
        scoreCollection[modelName + " "+ subj] = scoresArray
    dfScore = pd.DataFrame.from_dict(scoreCollection, orient='index', columns=EVALUATION_TOOLS + [TOXICITY, HONEST, REGARD])
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