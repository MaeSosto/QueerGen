from lib.constants import *
modelName = "llama3"
inputFilePath = OUTPUT_SENTENCES + modelName + '.csv'
templateFile = pd.read_csv(inputFilePath)

_tmp = []
for idx, row in templateFile.iterrows():
    _tmp.append(re.sub(row.loc[TEMPLATE], "", row.loc[GENERATED]))
    
templateFile[GENERATED] = _tmp
smallPart = pd.DataFrame.from_dict(templateFile)    
print(smallPart)
smallPart.to_csv(OUTPUT_SENTENCES+modelName+'.csv')