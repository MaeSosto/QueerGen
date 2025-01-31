from lib.constants import * 
from collections import defaultdict

folderPath = OUTPUT_EVAL_COM
files = [f for f in os.listdir(folderPath)]
for file in files:
    print(file)
    # modelName = file.replace('.csv', '')
    # df = pd.read_csv(file)

df = defaultdict(dict)
df['a']['b'] = 0
df['a']['c'] = 1
df['d']['c'] = 2
df['d']['b'] = 3
print(df)  
df = pd.DataFrame.from_dict(df)
print(df)