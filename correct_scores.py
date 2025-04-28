import pandas as pd   
from lib.constants import *
models = MODEL_LIST_FULL
for mod in models:
    data = pd.read_csv(f"{OUTPUT_EVALUATION+mod}.csv") 
    if AFINN in data:  
        data.drop(AFINN, inplace=True, axis=1) 
        data.to_csv(f"{OUTPUT_EVALUATION+mod}.csv", index=False)
