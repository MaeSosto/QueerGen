from lib.constants import *

# This Python function allows the user to choose a model from a list of options provided by the
# `MODEL_LIST` and returns the index of the chosen model.
# :return: The function `chooseModel()` returns the index of the chosen model from the
# `MODEL_LIST`.
# def chooseModel():
#     chosenModel = -1
#     while chosenModel < 0 or chosenModel > len(MODEL_LIST)-1:
#         print('๏ Select a model: ')
#         for idx, x in enumerate(MODEL_LIST):
#             print(f"[{idx}] -  {x}")
#         chosenModel = int(input())
#     return chosenModel

# The `clean_response` function removes newline characters, double quotes, and backticks from a given
# response string.
def clean_response(response):
    response = re.sub(r'\n', '', response)
    response = re.sub(r'\"', '', response)
    response = re.sub(r'`', '', response)
    response = response.replace('.', '')
    response = response.replace(r" '", "")
    response = response.replace(r"*", "")
    response = response.replace(r"[", "")
    response = response.replace(r"]", "")
    response = response.lower()
    response = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL)
    #response = re.sub(r'.', '', response)
    #response = f"['{response}']" 
    response = response.split(" ")
    response = response[-1]
    response = response.replace(r"is:", "")
    response = re.sub(r'[^a-zA-Z0-9]', '', response)
    #response = response.replace(r" ", "")
    return response

# The function `getListFromString` takes a string input, removes certain characters, splits the string
# by commas, and returns a list of the resulting elements.
def getListFromString(text):
    text = re.sub(r"'", "", str(text))
    text = re.sub(r'\]', '', text)
    text = re.sub(r'\[', '', text)
    return list(map(str, text.split(",")))

def getCSVFile(folder, modelName, predictionsConsidered):
    files = []
    for f in os.listdir(folder):
        pred = f.replace(f'{modelName}_', '').replace('.csv', '')
        try:
            if re.match(modelName, f) and int(pred) >= predictionsConsidered:
                files.append(int(pred))
        except: 
            continue
    files.sort()
    try:
        return pd.read_csv(f'{folder+modelName}_{files[0]}.csv')
    except Exception as X:
        print("EXC - There are no files related to the specified model [{modelName}] with at least {predictionsConsidered} words predicted")

# The function `getListFromString` takes a string containing a list of integers and returns a list of
# integers.
def getListFromString(text):
    text = re.sub(r"'", "", str(text))
    text = re.sub(r'\]', '', text)
    text = re.sub(r'\[', '', text)
    return (map(int, text.split(","))) if "," in text else text

def getTemplateFile(modelName, inputFolder, outputFolder):
    sentenceFile = f"{inputFolder+modelName}.csv"
    evaluationFile = f"{outputFolder+modelName}.csv"
    if os.path.exists(sentenceFile):
        sentenceFile = pd.read_csv(sentenceFile)
        #If the file exists already in the output folder then take that one   
        if os.path.exists(evaluationFile):
            evaluationFile = pd.read_csv(evaluationFile)
            print(f"๏ {evaluationFile.shape[0]} sentences imported!")
            return evaluationFile, sentenceFile[evaluationFile.shape[0]:]
        return pd.DataFrame(), sentenceFile[0:]
    else:
        return None, None