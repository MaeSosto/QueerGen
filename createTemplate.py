from lib.constants import *
TEMPLATE_PATH = DATA_SOURCE + 'template.csv'
NOUNS_PATH = DATA_SOURCE + 'nouns.csv'
TEMPLATE_PATH_COMPLETE = DATA_SOURCE + 'template_complete.csv'

def createTemplate():
    templateFile = pd.read_csv(TEMPLATE_PATH)
    dataList =[]
    print("๏ Generating template...")
    
    for _,row in tqdm(templateFile.iterrows(), total=templateFile.shape[0], desc='Creating template', unit=' s'):
        #Creating sentences with nouns and pronouns
        for _, r in pd.read_csv(NOUNS_PATH).iterrows():
            sentence = row.loc[TEMPLATE].replace(TARGET_, f"The {r.loc[SUBJECT]}" + f'{" person" if r.loc[PERSON] == "y" else ""}')
            
            dataList.append([
                row.loc[TEMPLATE],
                r.loc[SUBJECT], #subject
                sentence, #new template
                r.loc[TYPE], #type
                r.loc[CATEGORY] #category
            ]) 
    data_df = pd.DataFrame(dataList, columns=[ ORIGINAL, SUBJECT, TEMPLATE, TYPE, CATEGORY])
    data_df.to_csv(TEMPLATE_PATH_COMPLETE, index_label = 'index')
    print("๏ File template generated!")

createTemplate()
