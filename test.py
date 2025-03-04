import pandas as pd
from lib.constants import *
import matplotlib.pyplot as plt
from math import pi
import ast
import seaborn as sns

MODEL_NAME = GPT4_MINI

def stacked_bar_plot(model_name):
    # Load the CSV file
    df = pd.read_csv(f"{OUTPUT_EVALUATION+model_name}.csv")  # Replace with your actual file path
    #df.replace(pd.NA, "")
    
    for cat in HURTLEX_CATEGORIES_SHORT:
            df[HURTLEX + " "+ cat] = [0 for z in range(len(df))]
    for _,row in df.iterrows():
        cat = row.loc[HURTLEX]
        if str(cat) != "nan":
            row.loc[HURTLEX + " "+ cat] = row.loc[HURTLEX + " "+ cat] + 1
    
    print(df)
            
    mean_scores = df.groupby(TYPE)[[HURTLEX + " "+ cat for cat in HURTLEX_CATEGORIES_SHORT]].sum()

    print(mean_scores)
# Plot stacked bar chart
    mean_scores.plot(kind='bar', stacked=True, figsize=(10, 6), colormap='viridis', label= HURTLEX_CATEGORIES_SHORT)

    # Labels and Title
    plt.xlabel("Subject Type")
    plt.ylabel("Count")
    plt.title(f"Stacked Bar Chart of HurtLex Categories in {model_name}")
    plt.legend(HURTLEX_CATEGORIES_SHORT, title="HurtLex Categories")
    plt.xticks(rotation=0)

    # Show the plot
    plt.show()

stacked_bar_plot(MODEL_NAME)