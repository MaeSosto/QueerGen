from lib.constants import *
import pandas as pd
import matplotlib.pyplot as plt
import math
import seaborn as sns
import matplotlib.transforms as transforms
import random 
from statistics import mode
from wordcloud import WordCloud

PATH_SENTIMENT_GRAPH = OUTPUT_GRAPHS+'/sentiment/'
PATH_REGARD_GRAPH = OUTPUT_GRAPHS+'/regard/'
PATH_TOXICITY_GRAPH = OUTPUT_GRAPHS+'/toxicity/'
PATH_DIVERSITY_GRAPH = OUTPUT_GRAPHS+'/diversity/'
for path in [PATH_SENTIMENT_GRAPH, PATH_REGARD_GRAPH, PATH_TOXICITY_GRAPH, PATH_DIVERSITY_GRAPH]:
    os.makedirs(path, exist_ok=True) 
    
FONT_TITLE = 18
FONT_TICKS = 16
COLOR = 'color'
LINESTYLE = 'linestyle'
LABEL = 'label'
PATTERN = 'pattern'
IBM_COLORBLINDPALETTE = ['#ffb000', '#fe6100', '#dc267f', '#785ef0', '#648fff', '#000000']
MARKERS = ['o', "s", "^", "D", "X"]
patterns = [ "/" , "", ".","\\" , "|" , "-" , "+" , "x", "o", "O", "*"]

MODELS_GRAPHICS = {
    BERT_BASE : {
        LABEL: 'BERT Base',
        COLOR: '#ffb000', 
        LINESTYLE: '-',
        PATTERN: ""
    },
    BERT_LARGE : {
        LABEL: 'BERT Large',
        COLOR: '#ffb000', 
        LINESTYLE: '--',
        PATTERN: "/"
    },
    ROBERTA_BASE : {
        LABEL: 'RoBERTa Base',
        COLOR: '#fe6100', 
        LINESTYLE: '-',
        PATTERN: ""
    },
    ROBERTA_LARGE : {
        LABEL: 'RoBERTa Large',
        COLOR: '#fe6100', 
        LINESTYLE: '--',
        PATTERN: "/"
    },
    LLAMA3 : {
        LABEL: 'Llama 3',
        COLOR: '#648fff', 
        LINESTYLE: '-',
        PATTERN: ""
    },
    LLAMA3_70B : {
        LABEL: 'Llama 3 (70b)',
        COLOR: '#648fff', 
        LINESTYLE: '--',
        PATTERN: "/"
    },
    GEMMA3 : {
        LABEL: 'Gemma 3',
        COLOR: '#dc267f', 
        LINESTYLE: '-',
        PATTERN: ""
    },
    GEMMA3_27B : {
        LABEL: 'Gemma 3 (27b)',
        COLOR: '#dc267f', 
        LINESTYLE: '--',
        PATTERN: "/"
    },
    DEEPSEEK: {
        LABEL: 'DeepSeek R1',
        COLOR: '#785ef0', 
        LINESTYLE: '-',
        PATTERN: ""
    },
    DEEPSEEK_673B: {
        LABEL: 'DeepSeek R1 (671b)',
        COLOR: '#785ef0', 
        LINESTYLE: '--',
        PATTERN: "/"
    },
    GPT4_MINI : {
        LABEL: 'GPT4o Mini',
        COLOR: '#4ed22a', 
        LINESTYLE: '-',
        PATTERN: ""
    },
    GPT4 : {
        LABEL: 'GPT4o',
        COLOR: '#4ed22a', 
        LINESTYLE: '--',
        PATTERN: "/"
    },
    GEMINI_2_0_FLASH_LITE : {
        LABEL: "Gemini 2.0 Flash Lite",
        COLOR: '#196a49', 
        LINESTYLE: '-',
        PATTERN: ""
    },
    GEMINI_2_0_FLASH : {
        LABEL: "Gemini 2.0 Flash",
        COLOR: '#196a49', 
        LINESTYLE: '--',
        PATTERN: "/"
    }
}
    
PERSPECTIVE_MAP = {
    TOXICITY: { 
        LABEL: "Toxicity",
        COLOR: '#ffb000'
    }, 
    SEVERE_TOXICITY: { 
        LABEL: "Severe Toxicity",
        COLOR: '#fe6100'
    }, 
    INSULT: { 
        LABEL: "Insult",
        COLOR: '#dc267f'
    },  
    IDENTITY_ATTACK: { 
        LABEL: "Identity Attack",
        COLOR: '#785ef0'
    },  
    PROFANITY: { 
        LABEL: "Profanity",
        COLOR: '#648fff'
    }, 
    THREAT: { 
        LABEL: "Threat",
        COLOR: '#000000'
    }
}

def create_multi_line_graph(models, y_modelScores, y_label, path, img_name):
    fig, ax = plt.subplots(figsize=(12, 6))
    lines = []
    for idx, subjScore in enumerate(y_modelScores):
        label = MODELS_GRAPHICS[models[idx]].get(LABEL)
        color = MODELS_GRAPHICS[models[idx]].get(COLOR)
        marker = MARKERS[int(idx/2)]
        linestyle = MODELS_GRAPHICS[models[idx]].get(LINESTYLE)
        line, = ax.plot(SUBJ_CATEGORIES, subjScore, marker=marker, label=label, color=color, linestyle=linestyle)
        lines.append(line)

    # Set titles and labels
    ax.set_xlabel("Subject Category", fontsize=FONT_TICKS)
    ax.set_ylabel(y_label, fontsize=FONT_TICKS)
    ax.tick_params(axis='both', which='major', labelsize=FONT_TICKS)
    ax.grid(True, linestyle='--', alpha=0.25)

    # Create legend
    ax.legend(
        handles=lines,
        title="Model",
        title_fontsize=FONT_TICKS,
        fontsize=FONT_TICKS,
        loc='upper center',
        bbox_to_anchor=(0.5, -0.15),
        ncol = int(len(models)/2) if len(models) > 4 else 4 
    )

    plt.tight_layout()
    plt.savefig(path+img_name+'.png', bbox_inches = 'tight')
    
def heatmap_sentiment_analysis(models, TOOL):
    study = pd.DataFrame()
    y_modelScores =[]
    for model in models:
        data = pd.read_csv(f"{OUTPUT_EVALUATION+model}.csv") 
        subjScore =[]
        for subjCat in SUBJ_CATEGORIES:
            df = data[data[TYPE] == subjCat].copy()
            scoreMean = df[TOOL].dropna().tolist() #Get the scores for that subject category
            scoreMean = np.mean(scoreMean) #Calculate the mean for that subject category scores
            subjScore.append(scoreMean)
        y_modelScores.append(subjScore)
        study[model] = {cat: point for cat, point in zip(SUBJ_CATEGORIES, subjScore)}
    
    #display(study)
    heat = pd.DataFrame(study, index=SUBJ_CATEGORIES, columns=models)

    plt.figure(figsize=(16,6))  # Make it bigger horizontally and vertically

    ax =heatplot = sns.heatmap(
        heat,
        linewidths=1,
        annot=True,
        fmt=".3f",  
        cmap="RdYlGn",  # Red to Green colormap
        center=0,      # Center colormap at 0
        cbar_kws={'label': f'{TOOL} Score'},  # Colorbar label
        annot_kws={"size": FONT_TICKS}  # Annotation font size
    )
    ax.figure.axes[-1].yaxis.label.set_size(FONT_TICKS)

    # Set ticks and labels
    #plt.title("Sentiment Analysis Scores of the subject category's score averages accross models", fontsize = FONT_TITLE)
    ax.set_xlabel('Model', fontsize=FONT_TITLE)
    ax.set_ylabel('Subject Category', fontsize=FONT_TITLE)
    plt.xticks(rotation=45, rotation_mode="anchor", fontsize=FONT_TICKS)
    plt.yticks(rotation=45, rotation_mode="anchor", fontsize=FONT_TICKS)
    plt.setp(heatplot.xaxis.get_majorticklabels(), ha='right')
    plt.setp(heatplot.yaxis.get_majorticklabels(), ha='right')
    plt.tight_layout()
    #plt.show()
    plt.savefig(PATH_SENTIMENT_GRAPH+'heatmap.png', bbox_inches = 'tight')
