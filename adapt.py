from constants import * 

def adapt(model):
    inputPath = OUTPUT_PREDICTION+model+".csv"
    templateFile = pd.read_csv(inputPath)

    dicSentences = {
        'index' : [],
        TEMPLATE: [],
        TYPE: [],
        'category': [],
        'subject': [],
        'prediction': [],
    }
    for index,row in tqdm(templateFile.iterrows(), total=templateFile.shape[0], desc=f'Evaluating {model}\'s answers', unit=' sentences'):
        dicSentences['index'].append(index)
        dicSentences[TEMPLATE].append(row.loc[TEMPLATE])
        dicSentences[TYPE].append(row.loc[TYPE])
        dicSentences['category'].append("")
        dicSentences['subject'].append("")
        dicSentences['prediction'].append(f"['{re.sub(row.loc[TEMPLATE], '', row.loc[GENERATED].replace('.', ''))}']")
        df = pd.DataFrame.from_dict(dicSentences)    
        os.makedirs('adapted/', exist_ok=True)
        df.to_csv('adapted/'+model+".csv", sep=';', index=False)

adapt(LLAMA3)
            