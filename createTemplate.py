from lib.constants import * 

def createTemplate(templatePath, nounsPath):
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
                _sentence, #new template
                r.loc[TYPE], #type
                r.loc[CATEGORY], #category
                r.loc[SUBJECT] #subject
            ]) 

    data_df = pd.DataFrame(dataList, columns=[ "template", TYPE, CATEGORY, SUBJECT])
    data_df.to_csv(DATA_SOURCE + 'template_complete.csv', index_label = 'index')
    print("๏ File template generated!")
    

# The `createTemplate(templatePath, nounsPath)` function is reading data from two CSV files specified
# by `templatePath` and `nounsPath`. It then generates new templates by replacing placeholders in the
# sentences from the template file with values from the nouns file. The function iterates over each
# row in the template file and for each row, it iterates over the rows in the nouns file to create new
# sentences by replacing the placeholder with the subject from the nouns file.
templatePath, nounsPath = TEMPLATE_PATH, NOUNS_PATH
createTemplate(templatePath, nounsPath)
