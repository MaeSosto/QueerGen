from lib.constants import * 
from evaluate import load 


templateFile = pd.read_csv(OUTPUT_COMPLETE_SENTENCE+ 'llama3.csv')

predictions = templateFile[GENERATED]

perplexityTool = load("perplexity", module_type="metric")
perplexityList = perplexityTool.compute(predictions=predictions, model_id='gpt2')['perplexities']
templateFile[PERPLEXITY] = [truncate(per) for per in perplexityList]
#df = pd.DataFrame.from_dict(templateFile)    
templateFile.to_csv(OUTPUT_COMPLETE_SENTENCE+ 'llama3.csv')
