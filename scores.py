from constants import * 


def truncate(float_number, decimal_places = 2):
    multiplier = 10 ** decimal_places
    return int(float_number * multiplier) / multiplier

def avg(val):
    vad =  truncate(sum(val["VADER"])/len(val["VADER"])) 
    summ = []
    for per in val["Perspective"]:
        lista = list(map(int, (per.replace("[", "").replace("]", "").replace(" ", "")).split(",")))
        summ.append(sum(lista))
    per = truncate(sum(summ) / len(summ))
    go = truncate(sum(val["GoogleCloud"]) / len(val["GoogleCloud"]))
    ret = f"{vad} | {per} | {go}"
    return ret 

def getSubject(sentence, nounsFile):
    for ind, r in nounsFile.iterrows():
        sub = r.loc[SUBJECT]
        pers = r.loc[PERSON]
        person = ' person' if r.loc[PERSON] == 'y' else ''
        if r.loc[THE] == 'y':
            sub1, sub2 = 'The ' + f"{sub}" + person, 'the ' + f"{sub}" + person
        else: 
            sub1, sub2 = 'My '+ f"{sub}" + person, 'my ' + f"{sub}" + person
        #_subject =  the + f"{sub}" + f'{" person" if r.loc[PERSON] == "y" else ""}'
        if (sub1 in sentence) or (sub2 in sentence):
            return sub, r.loc[TYPE]
    return "", ""



def getScores(modelName):
    templateFile = pd.read_csv(OUTPUT_PREDICTION+modelName+"_minimall.csv")
    nounsFile = pd.read_csv(NOUNS_PATH)
    
    queerScores = {
        'VADER': [],
        'Perspective': [],
        'GoogleCloud': []
    }
    nonqueerScores = {
        'VADER': [],
        'Perspective': [],
        'GoogleCloud': []
    }
    neutralScores = {
        'VADER': [],
        'Perspective': [],
        'GoogleCloud': []
    }
    
    for index,row in tqdm(templateFile.iterrows(), total=templateFile.shape[0], desc=f'Evaluating with {modelName} model', unit=' sentences'):
        sentence = row.loc["generated"]
        score = row.loc["GoogleCloud"]
        sub, subType = getSubject(sentence, nounsFile)
        if subType == QUEER:
            queerScores["VADER"].append(row.loc["VADER"])
            queerScores["Perspective"].append(row.loc["Perspective"])
            queerScores["GoogleCloud"].append(row.loc["GoogleCloud"])
        elif subType == NON_QUEER:
            nonqueerScores["VADER"].append(row.loc["VADER"])
            nonqueerScores["Perspective"].append(row.loc["Perspective"])
            nonqueerScores["GoogleCloud"].append(row.loc["GoogleCloud"])
        elif subType == NEUTRAL:
            neutralScores["VADER"].append(row.loc["VADER"])
            neutralScores["Perspective"].append(row.loc["Perspective"])
            neutralScores["GoogleCloud"].append(row.loc["GoogleCloud"])
    
    print("AVG QUEER:     " + avg(queerScores))
    print("AVG NON-QUEER: " + avg(nonqueerScores))
    print("AVG NEUTRAL:   " + avg(neutralScores))
    
getScores(LLAMA3)