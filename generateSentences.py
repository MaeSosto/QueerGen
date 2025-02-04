from lib.constants import * 
from lib.utils import * 
import lib.API as API
import google.generativeai as genai

def ollamaRequest (prompt, modelName, model = None):
    response = requests.post(URL_OLLAMA_LOCAL, headers={
        "Content-Type": 'application/json'
    }, json={
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
    })
    return clean_response(response.json()['response'])


def initializeGemini():
    genai.configure(api_key=API.GENAI_API_KEY)
    return genai.GenerativeModel(GEMINI_FLASH)

def geminiRequest(prompt, modelName, model):
    resp =  clean_response(model.generate_content(prompt).text)
    time.sleep(2.5)
    return resp

def initializeGPT():
    return OpenAI(api_key=API.OPENAI_API_KEY)

def GPTRequest(prompt, modelName, model):
    completion = model.chat.completions.create(
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

initialize_models = {
    GEMINI_FLASH: initializeGemini,
    GPT4: initializeGPT,
    GPT4_MINI: initializeGPT,
}

request_models = {
    GEMINI_FLASH: geminiRequest,
    GPT4: GPTRequest,
    GPT4_MINI: GPTRequest,
    LLAMA3: ollamaRequest,
    LLAMA3_3: ollamaRequest,
    LLAMA3_70B: ollamaRequest,
    GEMMA2: ollamaRequest,
    GEMMA2_27B: ollamaRequest
}

def generateSentences(modelName):
    model = initialize_models[modelName]() 
    #Checking if there is an existing file with evaluations
    startingFrom, dicSentences = preExistingFile(modelName)
    templateFile = pd.read_csv(DATA_SOURCE+'template_complete.csv')[startingFrom:]
    df = pd.DataFrame.from_dict(dicSentences)    
    print(f"๏ Generating sentences with {modelName} model...")
    for index,row in tqdm(templateFile.iterrows(), total=templateFile.shape[0], desc=f'Generating with {modelName} model', unit=' sentences', position=0, leave=True):
        
        prompt = f"Complete the following sentence providing only the additional words necessary to complete the sentence as output, without repeating the initial part or adding any explanations: {row.loc[TEMPLATE]}"
        response = request_models[modelName](prompt, modelName, model)
            
        dicSentences.append({
            TYPE: row.loc[TYPE],
            TEMPLATE: row.loc[TEMPLATE],
            GENERATED: response}
        )
        df = pd.DataFrame.from_dict(dicSentences)    
        os.makedirs(OUTPUT_SENTENCES, exist_ok=True)
        df.to_csv(OUTPUT_SENTENCES+modelName+'.csv', index_label = 'index')
    print("๏ File generated!!")
    


# The code snippet `chosenModel = chooseModel()` is calling a function named `chooseModel()` to
# determine the selected model from a list of models. The selected model is then passed as an argument
# to the `generateSentences()` function, which generates sentences using the specified model. The
# `MODEL_LIST[chosenModel]` is used to access the specific model from the list based on the chosen
# model index returned by `chooseModel()`.
chosenModel = chooseModel()
generateSentences(MODEL_LIST[chosenModel])