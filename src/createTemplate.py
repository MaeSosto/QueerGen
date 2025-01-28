from lib.constants import * 

templateFile = pd.read_csv(TEMPLATE_PATH)
nounsFile = pd.read_csv(NOUNS_PATH)
dataList =[]
print("๏ Generating template...")

for templateNum, row in tqdm(templateFile.iterrows(), total=templateFile.shape[0], desc='Creating template', unit=' sentences'):
    sentence = row.loc[TEMPLATE]
    
    for ind, r in nounsFile.iterrows():
        init = True if sentence.split(TARGET_)[0] == '' else False
        the = ('The ' if init else 'the ') if r.loc[THE] == 'y' else ('My ' if init else 'my ')
        _subject =  the + f"{r.loc[SUBJECT]}" + f'{" person" if r.loc[PERSON] == "y" else ""}'
        _sentence = sentence.replace(TARGET_, _subject)
        
        dataList.append([
            row.loc['index'],
            _sentence, #new template
            r.loc[TYPE], #type
            r.loc[CATEGORY], #category
            r.loc[SUBJECT] #subject
        ]) 

data_df = pd.DataFrame(dataList, columns=[ "templateNum", "template", TYPE, CATEGORY, SUBJECT])
data_df.to_csv(TEMPLATES_COMPLETE_PATH, index_label = 'index')
print("๏ File template generated!")


