from lib.constants import * 
from evaluate import load 
from surprisal import AutoHuggingFaceModel
import numpy as np
import statistics as s

print("๏ Getting the CSV file...")
templateFile = pd.read_csv(OUTPUT_COMPLETE_SENTENCE+ 'llama3.csv')

predictions = templateFile[GENERATED]

print("๏ Calculating surprisal score...")
surprisalTool = AutoHuggingFaceModel.from_pretrained('gpt2')
surprisalList = surprisalTool.surprise(list(predictions))
surprisalScores = [truncate(s.mean([val for val in res.surprisals if val != np.inf])) for res in surprisalList]
    
print("๏ Calculating perplexity score...")
perplexityTool = load("perplexity", module_type="metric")
perplexityList = perplexityTool.compute(predictions=predictions, model_id='gpt2')['perplexities']
perplexityScores = [truncate(per) for per in perplexityList]


templateFile[PERPLEXITY] = perplexityScores
templateFile[SURPRISAL] = surprisalScores
#df = pd.DataFrame.from_dict(templateFile)    
templateFile.to_csv(OUTPUT_COMPLETE_SENTENCE+ 'llama3.csv')
print("๏ File updated correctly!")
