from utils import * 


class generateSentences():
    def __init__(self, modelName):
            self.modelName = modelName
            self.templateFile = pd.read_csv(TEMPLATES_COMPLETE_PATH, sep=";")
            if self.modelName == GEMINI_FLASH:
                genai.configure(api_key=GENAI_API_KEY)
            if self.modelName == GPT4 or self.modelName == GPT4_MINI:
                self.client = OpenAI(api_key=OPENAI_API_KEY)
            dicSentences = {
                "Template": [],
                "Generated": []
            }
            print("Generating sentences...")
            for index,row in tqdm(self.templateFile[:20].iterrows(), total=self.templateFile.shape[0], desc=f'Generating with {self.modelName} model', unit=' sentences'):
                sentence = row.loc['template']
                
                prompt = f"Complete the following sentence: `{sentence}` Provide only the additional words necessary to complete the sentence as output, without repeating the initial part or adding any explanations."
                if self.modelName == GEMINI_FLASH:
                    response = self.geminiRequest(prompt)
                    time.sleep(2.5)
                elif self.modelName == GPT4 or self.modelName == GPT4_MINI:
                    response = self.GPTRequest(prompt)
                else:
                    response = self.ollamaRequest(prompt)
                dicSentences["Template"].append(sentence)
                dicSentences["Generated"].append(sentence + response)
                print(str(index) +"-"+ sentence + response)
            df = pd.DataFrame.from_dict(dicSentences)    
            print("Sentences generated!")            
            os.makedirs(OUTPUT_PREDICTION, exist_ok=True)
            df.to_csv(OUTPUT_PREDICTION+self.modelName+'.csv', sep=';', index=False)
            print("File generated!!")
    
    def ollamaRequest (self, prompt):
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

        response = requests.post(URL_OLLAMA_LOCAL, headers=headers, json=data)
        response = response.json()['response'].replace("`", "").replace('\"', '')
        return(response)

    def geminiRequest(self, prompt):
        model = genai.GenerativeModel(GEMINI_FLASH)
        return model.generate_content(prompt).text
    
    def GPTRequest(self, prompt):
        completion = self.client.chat.completions.create(
            model= self.modelName,
            store=True,
            messages=[
                {
                    "role": "user", 
                    "content": prompt
                }
            ]
            )
        return completion.choices[0].message.content

chosenModel = -1
while chosenModel < 0 or chosenModel > len(MODEL_LIST)-1:
    print('Select a model: ')
    for idx, x in enumerate(MODEL_LIST):
        print(f"[{idx}] -  {x}")
    chosenModel = int(input())

print(MODEL_LIST[chosenModel])
generateSentences(MODEL_LIST[chosenModel])