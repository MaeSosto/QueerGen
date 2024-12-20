from utils import * 

class generateSentences():
    def __init__(self, modelName):
            self.modelName = modelName
            self.templateFile = pd.read_csv(TEMPLATES_COMPLETE_PATH, sep=";")
            dicSentences = {
                "Template": [],
                "Generated": []
            }
            print("Generating sentences...")
            for index,row in tqdm(self.templateFile.iterrows(), total=self.templateFile.shape[0], desc=f'Generating with {self.modelName} model', unit=' sentences'):
                sentence = row.loc['template']
                
                prompt = f"Complete the following sentence: `{sentence}` Provide only the additional words necessary to complete the sentence as output, without repeating the initial part or adding any explanations."
                response = self.modelRequest(prompt)
                dicSentences["Template"].append(sentence)
                dicSentences["Generated"].append(sentence + response)
                print(str(index) +"-"+ sentence + response)
            df = pd.DataFrame.from_dict(dicSentences)    
            print("Sentences generated!")            
            os.makedirs(OUTPUT_PREDICTION, exist_ok=True)
            df.to_csv(OUTPUT_PREDICTION+self.modelName+'.csv', sep=';', index=False)
            print("File generated!!")
    
    def modelRequest (self, prompt):
        data = {
            "model": self.modelName,
            "prompt": prompt,
            "messages": [
                {
                "role": "user",
                "content": prompt
                }
            ],   
            "stream": False
        }
        
        headers = {
            "Content-Type": 'application/json'
        }

        response = requests.post(url, headers=headers, json=data)
        response = response.json()['response'].replace("`", "").replace('\"', '')
        return(response)



chosenModel = -1
while chosenModel < 0 or chosenModel > len(MODEL_LIST)-1:
    print('Select a model: ')
    for idx, x in enumerate(MODEL_LIST):
        print(f"[{idx}] -  {x}")
    chosenModel = int(input())
    

#print(MODEL_LIST[chosenModel])
generateSentences(chosenModel)