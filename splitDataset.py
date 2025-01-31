from lib.constants import *
modelName = "llama3"
inputFilePath = OUTPUT_SENTENCES + modelName + '.csv'
templateFile = pd.read_csv(inputFilePath)

X = [row for idx, row in templateFile.iterrows()]
bigPart ,smallPart = train_test_split(X, test_size=0.3, random_state=42)

os.makedirs(OUTPUT_SENTENCES_SMALL, exist_ok=True)
smallPart = pd.DataFrame.from_dict(smallPart)    
smallPart.to_csv(OUTPUT_SENTENCES_SMALL+modelName+'.csv', index=False)