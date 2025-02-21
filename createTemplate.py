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
            init = True if row.loc[TEMPLATE].split(TARGET_)[0] == '' else False
            the = ('The ' if init else 'the ') if r.loc[THE] == 'y' else ('My ' if init else 'my ')
            _subject =  the + f"{r.loc[SUBJECT]}" + f'{" person" if r.loc[PERSON] == "y" else ""}'
            _sentence = row.loc[TEMPLATE].replace(TARGET_, _subject)
            
            dataList.append([
                row.loc[TEMPLATE],
                r.loc[SUBJECT], #subject
                _sentence, #new template
                r.loc[TYPE], #type
                r.loc[CATEGORY] #category
            ]) 
    data_df = pd.DataFrame(dataList, columns=[ ORIGINAL, SUBJECT, TEMPLATE, TYPE, CATEGORY])
    data_df.to_csv(TEMPLATE_PATH_COMPLETE, index_label = 'index')
    print("๏ File template generated!")

createTemplate()
