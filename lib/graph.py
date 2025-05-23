from lib.constants import *
import pandas as pd
import matplotlib.pyplot as plt
import math
import seaborn as sns
import matplotlib.transforms as transforms
import random 
from statistics import mode
from wordcloud import WordCloud

OUTPUT_GRAPHS = 'output_graphs/'
PATH_SENTIMENT_GRAPH = OUTPUT_GRAPHS+'/sentiment/'
PATH_REGARD_GRAPH = OUTPUT_GRAPHS+'/regard/'
PATH_TOXICITY_GRAPH = OUTPUT_GRAPHS+'/toxicity/'
PATH_DIVERSITY_GRAPH = OUTPUT_GRAPHS+'/diversity/'
for path in [PATH_SENTIMENT_GRAPH, PATH_REGARD_GRAPH, PATH_TOXICITY_GRAPH, PATH_DIVERSITY_GRAPH]:
    os.makedirs(path, exist_ok=True) 

FONT_TICKS = 22
FONT_LEGEND = FONT_TICKS# 18
COLOR = 'color'
LINESTYLE = 'linestyle'
LABEL = 'label'
PATTERN = 'pattern'
IBM_COLORBLINDPALETTE = ['#ffb000', '#fe6100', '#dc267f', '#785ef0', '#648fff', '#000000']
MARKERS = ['o', "s", "^", "D", "X","p", "v"]
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
    fig, ax = plt.subplots(figsize=(9, 5))
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
    # ax.legend(
    #     handles=lines,
    #     title="Model",
    #     title_fontsize=FONT_TICKS,
    #     fontsize=FONT_LEGEND,
    #     loc='upper center',
    #     bbox_to_anchor=(0.5, -0.25),
    #     ncol = 2 #int(len(models)/2) #if len(models) > 4 else 4 
    # )

    plt.tight_layout()
    plt.savefig(path+img_name+'.png', bbox_inches = 'tight')
    
def sentiment_analysis_heatmap(models, TOOL):
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
    #plt.title("Sentiment Analysis Scores of the subject category's score averages accross models", fontsize = FONT_TICKS)
    ax.set_xlabel('Model', fontsize=FONT_TICKS)
    ax.set_ylabel('Subject Category', fontsize=FONT_TICKS)
    plt.xticks(rotation=45, rotation_mode="anchor", fontsize=FONT_TICKS)
    plt.yticks(rotation=45, rotation_mode="anchor", fontsize=FONT_TICKS)
    plt.setp(heatplot.xaxis.get_majorticklabels(), ha='right')
    plt.setp(heatplot.yaxis.get_majorticklabels(), ha='right')
    plt.tight_layout()
    #plt.show()
    plt.savefig(PATH_SENTIMENT_GRAPH+'heatmap.png', bbox_inches = 'tight')



def diversity_bar(models):
    diversity_scores = []

    for model in models:
        data = pd.read_csv(f"{OUTPUT_EVALUATION + model}.csv") 
        for subjCat in SUBJ_CATEGORIES:
            df = data[data[TYPE] == subjCat].dropna(subset=[PREDICTION])

            if subjCat == UNMARKED:
                word_list = df[PREDICTION].tolist()
                diversity = round(len(set(word_list)) / len(df), 2)*100
            else:
                scores = [
                    round(len(set(sample[PREDICTION])) / len(sample), 2)*100
                    for _ in range(10)
                    for sample in [df.sample(n=100, replace=False, random_state=random.randint(0, 10000))]
                ]
                diversity = np.mean(scores)
            diversity_scores.append([model, subjCat, diversity])
    df_scores = pd.DataFrame(diversity_scores, columns=['Model', 'Subject Category', 'Diversity Score'])
    
    fig, ax = plt.subplots(figsize=(15, 5))
    num_models = len(models)
    total_group_width = 0.93 
    bar_width = total_group_width / num_models
    x_positions = np.arange(len(SUBJ_CATEGORIES))
    all_handles = []
    all_labels = []
    for i, model in enumerate(models):
        model_scores = df_scores[df_scores['Model'] == model]
        bars = ax.bar(x_positions + i * bar_width, 
                model_scores['Diversity Score'], 
                width=bar_width, 
                label=MODELS_GRAPHICS[model].get(LABEL), 
                color=MODELS_GRAPHICS[model].get(COLOR), 
                hatch=MODELS_GRAPHICS[model].get(PATTERN), 
                edgecolor='white',
                linewidth=2
            )
        all_handles.append(bars[0])
        all_labels.append(MODELS_GRAPHICS[model].get(LABEL))
        
    ax.set_xticks(x_positions + (total_group_width - bar_width) / 2)
    ax.set_xticklabels(SUBJ_CATEGORIES, fontsize=FONT_TICKS)
    ax.tick_params(axis='both', which='major', labelsize=FONT_TICKS)
    ax.set_xlabel("Subject Category", fontsize=FONT_TICKS)
    ax.set_ylabel("Diversity Score (%)", fontsize=FONT_TICKS)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    plt.ylim(top=80) #ymax is your value
    plt.ylim(bottom=20) #ymin is your value
    plt.subplots_adjust(bottom=0.25)
    fig.legend(
        all_handles,
        all_labels,
        title="Model",
        loc="center left",
        bbox_to_anchor=(0.9, 0.5),
        fancybox=True,
        shadow=True,
        ncol=1,  # stack vertically
        fontsize=FONT_LEGEND,
        title_fontsize=FONT_TICKS
    )

    plt.tight_layout(rect=[0, 0, 0.85, 1])  # leave space on the right for legend
    plt.savefig(PATH_DIVERSITY_GRAPH + 'diversity.png', bbox_inches='tight')

def most_common(lst, num):
    """Returns the `num` most common elements from `lst`."""
    topList, times = [], []
    num_tot_words = len(lst)
    for _ in range(num):
        if not lst:  # Prevent mode() from failing on empty lists
            break
        m = mode(lst)
        topList.append(m)
        m_times = int([lst.count(i) for i in set(lst) if i == m][0])
        times.append((m_times/num_tot_words)*100)
        lst = [l for l in lst if l != m]
    return topList, times


def diversity_wordCloud(models, num_top_words=3):
    fig, axes = plt.subplots(len(models), len(SUBJ_CATEGORIES), figsize=(4 * len(SUBJ_CATEGORIES), 2 * len(models)))
    axes = np.atleast_2d(axes)

    diversity_scores = []
    
    for row, model in enumerate(models):
        data = pd.read_csv(f"{OUTPUT_EVALUATION + model}.csv")
        for col, subjCat in enumerate(SUBJ_CATEGORIES):
            df = data[data[TYPE] == subjCat]
            word_list = df[PREDICTION].dropna().tolist()
            # Compute diversity score
            unique_words = set(word_list)
            diversity = (len(unique_words)/len(df))*100 if len(df) > 0 else 0
            top_words, num_times = most_common(word_list, num_top_words)
            if num_top_words == 1:
                diversity = round(diversity, 2)
                diversity_scores.append([MODELS_LABELS[model], subjCat, diversity, top_words[0], round(num_times[0], 2)])
            else:
                diversity_scores.append([MODELS_LABELS[model], subjCat, diversity, top_words, num_times])
            wordcloud = WordCloud(background_color="white", collocations=False).generate(" ".join(word_list))
            ax = axes[row, col]
            ax.imshow(wordcloud, interpolation="bilinear")
            ax.axis("off")
            if row == 0:
                ax.set_title(subjCat, fontsize=FONT_TICKS, fontweight = 'bold')
        
        fig.text(
            x=0.01,
            y=(len(models) - row - 0.5) / len(models),
            s=MODELS_LABELS[model],
            fontsize=FONT_TICKS,
            fontweight = 'bold',
            rotation=90,
            va="center",
            ha="center"
        )
    df_scores = pd.DataFrame(diversity_scores, columns=['Model', 'Subject Category', 'Diversity Score', 'Top Words', 'Occurrence'])
    df_scores.to_csv(PATH_DIVERSITY_GRAPH+"occurrenceDiversity.csv")

    plt.tight_layout(rect=[0.03, 0, 1, 1])
    plt.savefig(PATH_DIVERSITY_GRAPH + 'wordCloud.png', bbox_inches='tight')