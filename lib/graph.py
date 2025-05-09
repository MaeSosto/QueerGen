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

FONT_LEGEND = 18
FONT_TICKS = 20
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
    fig, ax = plt.subplots(figsize=(13, 6))
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
        fontsize=FONT_LEGEND,
        loc='upper center',
        bbox_to_anchor=(0.5, -0.20),
        ncol = int(len(models)/2) #if len(models) > 4 else 4 
    )

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

def regard_bar_graph(models, img_name, models_per_row):            
    study = pd.DataFrame()
    y_points_list = []

    for model in models:
        data = pd.read_csv(f"{OUTPUT_EVALUATION+model}.csv") 
        y_points = []
        for subjCat in SUBJ_CATEGORIES:
            df = data[data[TYPE] == subjCat].copy()
            regardScores = []
            for cat in REGARD_CATEGORIES:
                scoreList = df["Regard " + cat].dropna().tolist()
                scoreList = np.mean(scoreList)
                regardScores.append(scoreList)
            scoresSum = sum(regardScores)
            regardScores = [reg / scoresSum for reg in regardScores]
            y_points.append(regardScores)
        y_points_list.append(y_points)
        study[model] = {cat: point for cat, point in zip(SUBJ_CATEGORIES, y_points)}
    n_models = study.shape[1]
    num_cols = min(models_per_row, n_models)  
    num_rows = math.ceil(n_models / num_cols)

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(5 * num_cols, 5 * num_rows), sharey=True)
    axes = np.array(axes).flatten()

    if n_models == 1:
        axes = [axes]

    for idx, (ax, (model_name, category_data)) in enumerate(zip(axes, study.items())):
        bar_positions = np.linspace(0, len(SUBJ_CATEGORIES) - 1, len(SUBJ_CATEGORIES)) * 0.5
        bottoms = np.zeros(len(SUBJ_CATEGORIES))

        for stack_idx in range(len(REGARD_CATEGORIES)):
            heights = [category_data[subj][stack_idx] for subj in SUBJ_CATEGORIES]
            ax.bar(
                bar_positions,
                heights,
                bottom=bottoms,
                width=0.4,
                color=IBM_COLORBLINDPALETTE[stack_idx],
                label=REGARD_CATEGORIES[stack_idx],
                hatch=patterns[stack_idx],
                edgecolor='white',
                linewidth=2
            )
            bottoms += heights

        ax.set_title(MODELS_LABELS[model_name], fontsize=FONT_TICKS, fontweight="bold")
        ax.set_xticks(bar_positions)
        ax.set_xticklabels(SUBJ_CATEGORIES)
        ax.tick_params(axis='both', which='major', labelsize=FONT_TICKS)
        ax.grid(visible=True, axis='y', linestyle='--', alpha=0.7)

        if (len(models) - idx - models_per_row) <= 0:        
            ax.set_xlabel("Subject Category", size=FONT_TICKS)
        if (idx % models_per_row) == 0:
            ax.set_ylabel("Regard Score", size=FONT_TICKS)

    # Remove unused axes
    for j in range(n_models, len(axes)):
        fig.delaxes(axes[j])

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, title="Regard", title_fontsize=FONT_TICKS, fontsize=FONT_TICKS, loc='upper center', bbox_to_anchor=(0.5, 0), ncol=len(REGARD_CATEGORIES))

    plt.tight_layout()
    plt.savefig(PATH_REGARD_GRAPH + img_name + '.png', bbox_inches='tight')
    
def toxicity_markers_bar_chart(models, img_name):
    MARKER_COLOR_MAP = {
        UNMARKED: "black",
        NONQUEER: "dodgerblue",
        QUEER: "crimson"
    }
    tempComp = pd.read_csv(DATA_SOURCE + 'template_complete.csv')
    subject_type = tempComp.set_index(MARKER)[TYPE].to_dict()
    
    df_list = []
    for model in models:
        df = pd.read_csv(f"{OUTPUT_EVALUATION + model}.csv")
        for cat in PERSPECTIVE_CATEGORIES:
            df[cat] = (df[PERSPECTIVE + " " + cat]).astype(int)
        df_list.append(df)
    combined_df = pd.concat(df_list)

    sample_counts = combined_df.groupby(MARKER).size()
    binary_sums = combined_df.groupby(MARKER)[ [cat for cat in PERSPECTIVE_CATEGORIES] ].sum()
    binary_sums["perspective_total"] = binary_sums[PERSPECTIVE_CATEGORIES].sum(axis=1)
    percentage_df = binary_sums.div(sample_counts, axis=0).multiply(100)
    percentage_df = percentage_df.sort_values("perspective_total", ascending=True)

    # Plot
    fig, ax = plt.subplots(figsize=(16, 7))
    bar_positions = np.arange(len(percentage_df))
    bar_width = 0.8
    bottom = np.zeros(len(percentage_df))

    # Plot stacked bars
    for cat in PERSPECTIVE_CATEGORIES:
        values = percentage_df[cat]
        color = PERSPECTIVE_MAP[cat].get(COLOR)
        label = PERSPECTIVE_MAP[cat].get(LABEL)
        ax.bar(bar_positions, values, bar_width, bottom=bottom, label=label, color=color, edgecolor='white', linewidth=0.5)
        bottom += values.values

    unmarked_total = percentage_df.loc["Unmarked", "perspective_total"]
    if unmarked_total is not None:
        ax.axhline(unmarked_total, color='black', linestyle='--', linewidth=1, label="Unmarked")
        ax.text(2, unmarked_total + 1, "Unmarked", ha='right', va='top', fontsize=FONT_TICKS, color='black')
        
    ax.set_xlabel("Markers", fontsize=FONT_TICKS)
    ax.set_ylabel("Perspective Scores (%)", fontsize=FONT_TICKS)
    ax.tick_params(axis='both', which='major', labelsize=FONT_TICKS)
    ax.set_xticks(bar_positions)
    ax.set_xticklabels(percentage_df.index, rotation=45, ha="right")

    for tick, label in zip(ax.get_xticks(), ax.get_xticklabels()):
        marker = label.get_text()
        marker_type = subject_type.get(marker, None)
        if marker_type:
            label.set_color(MARKER_COLOR_MAP.get(marker_type, 'black'))

    ax.grid(visible=True, axis='y', linestyle='--', alpha=0.7)
    handles = [plt.Rectangle((0,0),1,1, color=PERSPECTIVE_MAP[cat].get(COLOR)) for cat in PERSPECTIVE_CATEGORIES]
    labels = [PERSPECTIVE_MAP[cat].get(LABEL) for cat in PERSPECTIVE_CATEGORIES] 
    title = "Perpective API Category"
    fig.legend(handles, labels, title=title, loc="center left", bbox_to_anchor=(1.01, 0.7),
             borderaxespad=0, fontsize=FONT_LEGEND, title_fontsize=FONT_TICKS)
    #fig.legend(handles, labels, title=title,  title_fontsize=FONT_TICKS, fontsize=FONT_LEGEND,loc='upper center', bbox_to_anchor=(0.5, 0), ncol=len(PERSPECTIVE_CATEGORIES))

    plt.tight_layout()
    plt.savefig(PATH_TOXICITY_GRAPH + img_name + '.png', bbox_inches='tight')

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
    
def diversity_wordCloud(models):
    fig, axes = plt.subplots(len(models), len(SUBJ_CATEGORIES), figsize=(4 * len(SUBJ_CATEGORIES), 2 * len(models)))
    axes = np.atleast_2d(axes)

    for row, model in enumerate(models):
        data = pd.read_csv(f"{OUTPUT_EVALUATION + model}.csv")
        for col, subjCat in enumerate(SUBJ_CATEGORIES):
            df = data[data[TYPE] == subjCat]
            word_list = df[PREDICTION].dropna().tolist()
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

    plt.tight_layout(rect=[0.03, 0, 1, 1])
    plt.savefig(PATH_DIVERSITY_GRAPH + 'wordCloud.png', bbox_inches='tight')