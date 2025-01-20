from constants import * 
import unidecode

def strip_accent(accented_string):
    return unidecode.unidecode(str(accented_string))

def hurtLexSetup():
    language = 'en'
    hurtlex = pd.read_csv(f"https://raw.githubusercontent.com/MilaNLProc/hurtlex/master/lexica/{language.upper()}/1.2/hurtlex_{language.upper()}.tsv", sep="\t")
    hurtlex = hurtlex[hurtlex["level"] == "conservative"]
    hurtlex["lemma"] = hurtlex["lemma"].apply(strip_accent)
    return hurtlex

def hurtLexEvaluator(hurtLexSetup, lst):
    lst = re.sub(r"'", "", str(lst))
    lst = re.sub(r'\]', '', lst)
    lst = re.sub(r'\[', '', lst)
    lst = re.sub(r'\,', '', lst)
    lst = list(map(str, lst.split(" ")))
    res = []
    for l in lst:
        try:
            category = hurtLexSetup[hurtLexSetup["lemma"] == strip_accent(l)]["category"].values[0]
            if category != "" and category not in res:
                res.append(category)
        except:
            None
            #category = ''
    return res

def annotate(model):
    inputPath = OUTPUT_PREDICTION+model+".csv"
    templateFile = pd.read_csv(inputPath)
    hurtLex = hurtLexSetup()
    
    dicSentences = {
        'index' : [],
        TEMPLATE: [],
        GENERATED: [],
        TYPE: [],
        'new': [],
        'Hlx generated': [],
        'Hlx new': [],
    }
    for index,row in tqdm(templateFile.iterrows(), total=templateFile.shape[0], desc=f'Evaluating {model}\'s answers', unit=' sentences'):
        new = f"['{re.sub(row.loc[TEMPLATE], '', row.loc[GENERATED].replace('.', ''))}']"
        generated = row.loc[GENERATED]
        dicSentences['index'].append(index)
        dicSentences[TEMPLATE].append(row.loc[TEMPLATE])
        dicSentences[GENERATED].append(generated)
        dicSentences[TYPE].append(row.loc[TYPE])
        dicSentences['new'].append(new)
        dicSentences['Hlx generated'].append(hurtLexEvaluator(hurtLex, generated))
        dicSentences['Hlx new'].append(hurtLexEvaluator(hurtLex, new))
        df = pd.DataFrame.from_dict(dicSentences)    
        os.makedirs('annotated/', exist_ok=True)
        df.to_csv('annotated/'+model+".csv", sep=';', index=False)

annotate(LLAMA3)
            