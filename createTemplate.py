from utils import * 

# # Build Complete Template
class CompleteTemplateBuilder():
    def __init__(self):
        self.template = pd.read_csv(TEMPLATE_PATH)
        self.nouns = pd.read_csv(NOUNS_PATH)
        self.template_builder()

    def plural_form(self, be, sentence):
        if be == 'are':
            word = sentence.split(" ")[1]
            if word[-1] == 's':
                sentence = sentence.replace(word, word[:-1])
        return sentence

    def template_builder(self):
        dataList =[]
        for templateNum, row in tqdm(self.template.iterrows(), total=self.template.shape[0], desc='Creating template', unit=' sentences'):
            sentence = row.loc[TEMPLATE]
            
            #Creating sentences with nouns
            for ind, r in self.nouns.iterrows():
                init = True if sentence.split(TARGET_)[0] == '' else False
                the = ('The ' if init else 'the ') if r.loc[THE] == 'y' else ('My ' if init else 'my ')
                _subject =  the + f"{r.loc[SUBJECT]}" + f'{" person" if r.loc[PERSON] == 'y' else ""}'
                _sentence = sentence.replace(TARGET_, _subject)
                _sentence = _sentence.replace(BE_, 'is').replace(WERE_, 'was').replace(HAVE_, 'has')
                
                dataList.append([
                    templateNum,
                    _sentence, #new template
                    r.loc[TYPE], #type
                    r.loc[CATEGORY], #category
                    r.loc[SUBJECT] #subject
                ]) 

        data_df = pd.DataFrame(dataList, columns=[ "templateNum","template", TYPE, CATEGORY, SUBJECT])
        #display(data_df)
        print(data_df)
        os.makedirs(OUTPUT_TEMPLATE, exist_ok=True)
        data_df.to_csv(TEMPLATES_COMPLETE_PATH, index_label = 'index')

CompleteTemplateBuilder()

