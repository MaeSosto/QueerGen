from lib.constants import *
TEMPLATE_PATH = DATA_SOURCE + 'template.csv'
NOUNS_PATH = DATA_SOURCE + 'nouns.csv'
TEMPLATE_PATH_COMPLETE = DATA_SOURCE + 'template_complete.csv'

def createTemplate():
    templateFile = pd.read_csv(TEMPLATE_PATH)
    dataList =[]
    print("๏ Generating template...")
    
    for _,row in tqdm(templateFile.iterrows(), total=templateFile.shape[0], desc='Creating template', unit=' s'):
        nounsFile = pd.read_csv(NOUNS_PATH)
        neutralList = nounsFile[nounsFile[TYPE] == NEUTRAL].copy()
        #neutralList = neutralList[SUBJECT].tolist()
        nonNeutralList = nounsFile[nounsFile[TYPE] != NEUTRAL].copy()
        
        template = row.loc[TEMPLATE]
        #Creating sentences with nouns
        for _, neu in neutralList.iterrows():
            dataList.append([
                template, #original
                neu.loc[SUBJECT], #subject
                template.replace(TARGET_, f"The {neu.loc[SUBJECT]}"), #template
                neu.loc[TYPE], #type
                neu.loc[CATEGORY] #category
            ])    
            for _,r in nonNeutralList.iterrows():
                if r.loc[PERSON] == "y":
                    subject = f"The {r.loc[SUBJECT]} {neu.loc[SUBJECT]}"
                elif r.loc[PERSON] == "n" and neu.loc[SUBJECT] == 'person':
                    subject = f"The {r.loc[SUBJECT]}"
                else:
                    subject = f"The {neu.loc[SUBJECT]} is a {r.loc[SUBJECT]} and"
                sentence = template.replace(TARGET_, subject)
                
                dataList.append([
                    template, #original
                    r.loc[SUBJECT], #subject
                    sentence, #template
                    r.loc[TYPE], #type
                    r.loc[CATEGORY] #category
                ]) 
    data_df = pd.DataFrame(dataList, columns=[ ORIGINAL, SUBJECT, TEMPLATE, TYPE, CATEGORY])
    data_df.to_csv(TEMPLATE_PATH_COMPLETE, index_label = 'index')
    print("๏ File template generated!")

createTemplate()
