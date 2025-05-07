import pandas as pd   
from lib.constants import *
models = MODEL_LIST_FULL
for mod in models:
    data = pd.read_csv(f"{OUTPUT_EVALUATION+mod}.csv") 
    for reg in REGARD_CATEGORIES:
        if REGARD + " "+ reg in data:  
            data.drop(REGARD + " "+ reg, inplace=True, axis=1) 
            data.to_csv(f"{OUTPUT_EVALUATION+mod}.csv", index=False)
