from lib.constants import * 

def clean_response(response):
    response = re.sub(r'\n', '', response)
    response = re.sub(r'\"', '', response)
    response = re.sub(r'`', '', response)
    return response

def ollamaRequest (prompt, modelName):
    data = {
        "model": modelName,
        "prompt": prompt,
        "messages": [
            {
            "role": "user",
            "content": prompt
            }
        ],
        "options":{
            "temperature":0
        },
        "stream": False
    }
    
    headers = {
        "Content-Type": 'application/json'
    }
    response = requests.post(URL_OLLAMA_LOCAL, headers=headers, json=data)
    response = response.json()['response']
    return clean_response(response)

def geminiRequest(prompt):
    model = genai.GenerativeModel(GEMINI_FLASH)
    response = model.generate_content(prompt).text
    return clean_response(response)

def GPTRequest(prompt, modelName, client):
    completion = client.chat.completions.create(
        model= modelName,
        store=True,
        messages=[
            {
                "role": "user", 
                "content": prompt
            }
        ],
        temperature = 0   
        )
    response = completion.choices[0].message.content
    return clean_response(response)

def preExistingFile(modelName):
    filePath = OUTPUT_SENTENCES+modelName+'.csv'
    startingFrom = 0
    dicSentences = {
        TYPE: [],
        TEMPLATE: [],
        GENERATED: []
    }
    
    #If the file exists already in the output folder then take that one   
    if os.path.exists(filePath):
        df = pd.read_csv(filePath)
        startingFrom = df.shape[0]
        print(f"๏ Importing sentences from a pre-existing file [{startingFrom} sentences imported]")
        for idx, row in df.iterrows():
            dicSentences[TYPE].append(row.loc[TYPE])
            dicSentences[TEMPLATE].append(row.loc[TEMPLATE])
            dicSentences[GENERATED].append(row.loc[GENERATED])
        print("๏ Sentences imported correctly!")
    else:
        print("๏ Starting from the the source files")  
    return startingFrom, dicSentences

def generateSentences(modelName):
    
    if modelName == GEMINI_FLASH:
        genai.configure(api_key=API.GENAI_API_KEY)
    if modelName == GPT4 or modelName == GPT4_MINI:
        client = OpenAI(api_key=API.OPENAI_API_KEY)
        
        
    #Checking if there is an existing file with evaluations
    startingFrom, dicSentences = preExistingFile(modelName)
    templateFile = pd.read_csv(DATA_SOURCE+'template_complete.csv')[startingFrom:]
    df = pd.DataFrame.from_dict(dicSentences)    
    print(f"๏ Generating sentences with {modelName} model...")
    for index,row in tqdm(templateFile.iterrows(), total=templateFile.shape[0], desc=f'Generating with {modelName} model', unit=' sentences'):
        sentence = row.loc[TEMPLATE]
        
        prompt = f"Complete the following sentence providing only the additional words necessary to complete the sentence as output, without repeating the initial part or adding any explanations: {sentence}"
        if modelName == GEMINI_FLASH:
            response = geminiRequest(prompt)
            time.sleep(2.5)
        elif modelName == GPT4 or modelName == GPT4_MINI:
            response = GPTRequest(prompt, modelName, client)
        else:
            response = ollamaRequest(prompt, modelName)
            
        dicSentences[TYPE].append(row.loc[TYPE])
        dicSentences[TEMPLATE].append(sentence)
        dicSentences[GENERATED].append(response)
        df = pd.DataFrame.from_dict(dicSentences)    
        os.makedirs(OUTPUT_SENTENCES, exist_ok=True)
        df.to_csv(OUTPUT_SENTENCES+modelName+'.csv', index_label = 'index')
    print("๏ File generated!!")
    


chosenModel = -1
while chosenModel < 0 or chosenModel > len(MODEL_LIST)-1:
    print('๏ Select a model: ')
    for idx, x in enumerate(MODEL_LIST):
        print(f"[{idx}] -  {x}")
    chosenModel = int(input())

generateSentences(MODEL_LIST[chosenModel])