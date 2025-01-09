from constants import * 
import support_lib
import API

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
        "stream": False
    }
    
    headers = {
        "Content-Type": 'application/json'
    }
    response = requests.post(URL_OLLAMA_LOCAL, headers=headers, json=data)
    response = response.json()['response']
    return support_lib.clean_response(response)

def geminiRequest(prompt):
    model = genai.GenerativeModel(GEMINI_FLASH)
    response = model.generate_content(prompt).text
    return support_lib.clean_response(response)

def GPTRequest(prompt, modelName, client):
    completion = client.chat.completions.create(
        model= modelName,
        store=True,
        messages=[
            {
                "role": "user", 
                "content": prompt
            }
        ]
        )
    response = completion.choices[0].message.content
    return support_lib.clean_response(response)

def generateSentences(modelName):
    templateFile = pd.read_csv(TEMPLATES_COMPLETE_PATH)
    if modelName == GEMINI_FLASH:
        genai.configure(api_key=API.GENAI_API_KEY)
    if modelName == GPT4 or modelName == GPT4_MINI:
        client = OpenAI(api_key=API.OPENAI_API_KEY)
    dicSentences = {
        TEMPLATE: [],
        GENERATED: []
    }
    print(f"๏ Generating sentences with {modelName} model...")
    #templateFile = templateFile[11000:11200]
    for index,row in tqdm(templateFile.iterrows(), total=templateFile.shape[0], desc=f'Generating with {modelName} model', unit=' sentences'):
        sentence = row.loc[TEMPLATE]
        
        prompt = f"Complete the following sentence providing only the additional words necessary to complete the sentence as output, without repeating the initial part or adding any explanations: : {sentence}"
        if modelName == GEMINI_FLASH:
            response = geminiRequest(prompt)
            time.sleep(2.5)
        elif modelName == GPT4 or modelName == GPT4_MINI:
            response = GPTRequest(prompt, modelName, client)
        else:
            response = ollamaRequest(prompt, modelName)
        dicSentences[TEMPLATE].append(sentence)
        dicSentences[GENERATED].append(sentence + response)
        #print(str(index) +"-"+ sentence + response)
    df = pd.DataFrame.from_dict(dicSentences)    
    print("๏ Sentences generated!")            
    os.makedirs(OUTPUT_PREDICTION, exist_ok=True)
    df.to_csv(OUTPUT_PREDICTION+modelName+'_minimal.csv', index_label = 'index')
    print("๏ File generated!!")
    


chosenModel = -1
while chosenModel < 0 or chosenModel > len(MODEL_LIST)-1:
    print('๏ Select a model: ')
    for idx, x in enumerate(MODEL_LIST):
        print(f"[{idx}] -  {x}")
    chosenModel = int(input())

generateSentences(MODEL_LIST[chosenModel])