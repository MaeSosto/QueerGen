# === Imports ===
from lib import *
import math
import random
import matplotlib.pyplot as plt

# === Constants and Configuration ===
OUTPUT_GRAPHS = 'output_graphs/'
PATH_SENTIMENT_GRAPH = os.path.join(OUTPUT_GRAPHS, 'sentiment/')
PATH_REGARD_GRAPH = os.path.join(OUTPUT_GRAPHS, 'regard/')
PATH_TOXICITY_GRAPH = os.path.join(OUTPUT_GRAPHS, 'toxicity/')
PATH_DIVERSITY_GRAPH = os.path.join(OUTPUT_GRAPHS, 'diversity/')

for path in [OUTPUT_GRAPHS, PATH_SENTIMENT_GRAPH, PATH_REGARD_GRAPH, PATH_TOXICITY_GRAPH, PATH_DIVERSITY_GRAPH]:
    os.makedirs(path, exist_ok=True)

COLOR = 'color'
LINESTYLE = 'linestyle'
LABEL = 'label'
PATTERN = 'pattern'
IBM_COLORBLINDPALETTE = ['#ffb000', '#fe6100', '#dc267f', '#785ef0', '#648fff', '#000000']
MARKERS = ['o', "s", "^", "D", "X","p", "v"]
patterns = ["/", "", ".", "\\", "|", "-", "+", "x", "o", "O", "*"]

MLM_MODELS = "bertModels"
OPEN_MODELS = "openModels"
CLOSE_MODELS = "closeModels"

# === Model Visualization Metadata ===
MODELS_GRAPHICS = {
    BERT_BASE:       {LABEL: 'BERT Base', COLOR: '#ffb000', LINESTYLE: '-',  PATTERN: ""},
    BERT_LARGE:      {LABEL: 'BERT Large', COLOR: '#ffb000', LINESTYLE: '--', PATTERN: "/"},
    ROBERTA_BASE:    {LABEL: 'RoBERTa Base', COLOR: '#fe6100', LINESTYLE: '-',  PATTERN: ""},
    ROBERTA_LARGE:   {LABEL: 'RoBERTa Large', COLOR: '#fe6100', LINESTYLE: '--', PATTERN: "/"},
    LLAMA3:          {LABEL: 'Llama 3', COLOR: '#648fff', LINESTYLE: '-',  PATTERN: ""},
    LLAMA3_70B:      {LABEL: 'Llama 3 (70b)', COLOR: '#648fff', LINESTYLE: '--', PATTERN: "/"},
    GEMMA3:          {LABEL: 'Gemma 3', COLOR: '#dc267f', LINESTYLE: '-',  PATTERN: ""},
    GEMMA3_27B:      {LABEL: 'Gemma 3 (27b)', COLOR: '#dc267f', LINESTYLE: '--', PATTERN: "/"},
    DEEPSEEK:        {LABEL: 'DeepSeek R1', COLOR: '#785ef0', LINESTYLE: '-',  PATTERN: ""},
    DEEPSEEK_671B:   {LABEL: 'DeepSeek R1 (671b)', COLOR: '#785ef0', LINESTYLE: '--', PATTERN: "/"},
    GPT4_MINI:       {LABEL: 'GPT4o Mini', COLOR: '#4ed22a', LINESTYLE: '-',  PATTERN: ""},
    GPT4:            {LABEL: 'GPT4o', COLOR: '#4ed22a', LINESTYLE: '--', PATTERN: "/"},
    GEMINI_2_0_FLASH_LITE: {LABEL: "Gemini 2.0 Flash Lite", COLOR: '#196a49', LINESTYLE: '-', PATTERN: ""},
    GEMINI_2_0_FLASH:     {LABEL: "Gemini 2.0 Flash", COLOR: '#196a49', LINESTYLE: '--', PATTERN: "/"},
}

# === Perspective API Categories ===
PERSPECTIVE_MAP = {
    TOXICITY:         {LABEL: "Toxicity", COLOR: '#ffb000'},
    SEVERE_TOXICITY:  {LABEL: "Severe Toxicity", COLOR: '#fe6100'},
    INSULT:           {LABEL: "Insult", COLOR: '#dc267f'},
    IDENTITY_ATTACK:  {LABEL: "Identity Attack", COLOR: '#785ef0'},
    PROFANITY:        {LABEL: "Profanity", COLOR: '#648fff'},
    THREAT:           {LABEL: "Threat", COLOR: '#000000'},
}

def toxicity_markers_bar_chart(models, img_name):
    font_size = 18
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
            df[cat] = df[f"{PERSPECTIVE} {cat}"].astype(int)
        df_list.append(df)

    combined_df = pd.concat(df_list)
    sample_counts = combined_df.groupby(MARKER).size()
    binary_sums = combined_df.groupby(MARKER)[PERSPECTIVE_CATEGORIES].sum()
    binary_sums["perspective_total"] = binary_sums.sum(axis=1)
    percentage_df = binary_sums.div(sample_counts, axis=0).multiply(100)
    percentage_df = percentage_df.sort_values("perspective_total", ascending=False)

    fig, ax = plt.subplots(figsize=(16, 5))
    bar_positions = np.arange(len(percentage_df))
    bottom = np.zeros(len(percentage_df))

    for cat in PERSPECTIVE_CATEGORIES:
        values = percentage_df[cat]
        ax.bar(
            bar_positions,
            values,
            width=0.8,
            bottom=bottom,
            label=PERSPECTIVE_MAP[cat][LABEL],
            color=PERSPECTIVE_MAP[cat][COLOR],
            edgecolor='white',
            linewidth=0.5
        )
        bottom += values.values

    unmarked_total = percentage_df.loc["Unmarked", "perspective_total"]
    ax.axhline(unmarked_total, color='black', linestyle='--', linewidth=1, label="Unmarked")
    ax.text(30, unmarked_total + 1, "Unmarked", ha='right', va='top', fontsize=font_size, color='black')

    ax.set_ylabel("Persp. Average Toxicity (%)", fontsize=font_size)
    ax.tick_params(axis='both', labelsize=font_size)
    ax.set_xticks(bar_positions)
    ax.set_xticklabels(percentage_df.index, rotation=25, ha="right")

    for tick, label in zip(ax.get_xticks(), ax.get_xticklabels()):
        marker = label.get_text()
        marker_type = subject_type.get(marker)
        if marker_type:
            label.set_color(MARKER_COLOR_MAP.get(marker_type, 'black'))

    ax.grid(visible=True, axis='y', linestyle='--', alpha=0.7)
    handles = [plt.Rectangle((0,0),1,1, color=PERSPECTIVE_MAP[cat][COLOR]) for cat in PERSPECTIVE_CATEGORIES]
    labels = [PERSPECTIVE_MAP[cat][LABEL] for cat in PERSPECTIVE_CATEGORIES]

    fig.legend(handles, labels, title="Perspective API Category", loc="upper right", borderaxespad=0.2, fontsize=font_size, title_fontsize=font_size, ncol=2)
    plt.tight_layout()
    plt.savefig(PATH_TOXICITY_GRAPH + img_name + '.png', bbox_inches='tight')
    plt.close()


def regard_bar_graph(models, img_name, models_per_row):
    font_size = 16
    model_scores = {}

    for model in models:
        data = pd.read_csv(f"{OUTPUT_EVALUATION + model}.csv")
        subj_scores = []

        for subjCat in SUBJ_CATEGORIES:
            df = data[data[TYPE] == subjCat]
            raw_scores = [df[f"Regard {cat}"].dropna().mean() for cat in REGARD_CATEGORIES]
            score_sum = sum(raw_scores)
            normalized = [score / score_sum for score in raw_scores]
            subj_scores.append(normalized)

        model_scores[model] = subj_scores

    n_models = len(models)
    num_cols = min(models_per_row, n_models)
    num_rows = math.ceil(n_models / num_cols)
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(16, 6), sharey=True)
    axes = np.array(axes).flatten() if n_models > 1 else [axes]

    for idx, model in enumerate(models):
        ax = axes[idx]
        scores = model_scores[model]
        positions = np.linspace(0, len(SUBJ_CATEGORIES) - 1, len(SUBJ_CATEGORIES)) * 0.2
        bottom = np.zeros(len(SUBJ_CATEGORIES))

        for i, category in enumerate(REGARD_CATEGORIES):
            heights = [score[i] for score in scores]
            ax.bar(
                positions,
                heights,
                bottom=bottom,
                width=0.2,
                color=IBM_COLORBLINDPALETTE[i],
                hatch=patterns[i],
                edgecolor='white',
                linewidth=2,
                label=category
            )
            bottom += heights

        ax.set_title(MODELS_LABELS[model], fontsize=font_size)
        ax.set_xticks(positions)
        ax.set_xticklabels(["U", "NQ", "Q"])
        ax.tick_params(axis='both', labelsize=font_size)
        ax.grid(visible=True, axis='y', linestyle='--', alpha=0.7)
        if idx % models_per_row == 0:
            ax.set_ylabel("Regard Score", size=font_size)

    for j in range(n_models, len(axes)):
        fig.delaxes(axes[j])

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, title="Regard", title_fontsize=font_size, fontsize=font_size, loc='upper center', bbox_to_anchor=(0.5, 0), ncol=len(REGARD_CATEGORIES))
    plt.tight_layout()
    plt.savefig(PATH_REGARD_GRAPH + img_name + '.png', bbox_inches='tight')
    plt.close()

def create_multi_line_graph(models, scores_by_subject, y_label, path, img_name):
    font_size = 22
    fig, ax = plt.subplots(figsize=(9, 5))

    for idx, scores in enumerate(scores_by_subject):
        model = models[idx]
        style = MODELS_GRAPHICS[model]
        marker = MARKERS[int(idx / 2)]
        ax.plot(
            SUBJ_CATEGORIES,
            scores,
            label=style.get(LABEL),
            color=style.get(COLOR),
            linestyle=style.get(LINESTYLE),
            marker=marker
        )

    ax.set_xlabel("Subject Category", fontsize=font_size)
    ax.set_ylabel(y_label, fontsize=font_size)
    ax.tick_params(axis='both', labelsize=font_size)
    ax.grid(True, linestyle='--', alpha=0.25)
    ax.legend()
    plt.tight_layout()
    plt.savefig(f"{path}{img_name}.png", bbox_inches='tight')
    plt.close()


def sentiment_analysis_scores(models, img_name, tool=VADER, path=PATH_SENTIMENT_GRAPH):
    all_scores = []

    for model in models:
        data = pd.read_csv(f"{OUTPUT_EVALUATION}{model}.csv")
        subj_scores = []

        for category in SUBJ_CATEGORIES:
            df = data[data[TYPE] == category]
            scores = df[tool].dropna().tolist()
            subj_scores.append(np.mean(scores))

        all_scores.append(subj_scores)

    y_label = f"{tool} Scores [-1, 1]"
    create_multi_line_graph(models, all_scores, y_label, path, img_name)


def toxicity_scores(models, img_name, path=PATH_TOXICITY_GRAPH):
    all_scores = []

    for model in models:
        data = pd.read_csv(f"{OUTPUT_EVALUATION}{model}.csv")
        perspective_cols = [f"{PERSPECTIVE} {cat}" for cat in PERSPECTIVE_CATEGORIES]

        subj_scores = []
        for category in SUBJ_CATEGORIES:
            df = data[data[TYPE] == category]
            perspective_avg = df[perspective_cols].astype(int).sum(axis=1).mean() * 100
            subj_scores.append(perspective_avg)

        all_scores.append(subj_scores)

    y_label = "Persp. Avg Toxicity (%)"
    create_multi_line_graph(models, all_scores, y_label, path, img_name)


def diversity_score(models, img_name, path=PATH_DIVERSITY_GRAPH):
    all_scores = []

    for model in models:
        data = pd.read_csv(f"{OUTPUT_EVALUATION}{model}.csv")
        subj_scores = []

        for category in SUBJ_CATEGORIES:
            df = data[data[TYPE] == category].dropna(subset=[PREDICTION])

            if category == UNMARKED:
                diversity = round(len(set(df[PREDICTION])) / len(df) * 100, 2)
            else:
                diversities = [
                    round(len(set(sample[PREDICTION])) / len(sample) * 100, 2)
                    for _ in range(10)
                    for sample in [df.sample(n=100, replace=False, random_state=random.randint(0, 10000))]
                ]
                diversity = int(np.mean(diversities))

            subj_scores.append(diversity)

        all_scores.append(subj_scores)

    y_label = "Lexical Diversity (%)"
    create_multi_line_graph(models, all_scores, y_label, path, img_name)


# === Main Execution ===
for model_group, name in [
    (MODEL_MLM, MLM_MODELS),
    (MODEL_OPEN, OPEN_MODELS),
    (MODEL_CLOSE, CLOSE_MODELS)
]:
    sentiment_analysis_scores(model_group, name)
    toxicity_scores(model_group, name)
    diversity_score(model_group, name)

toxicity_markers_bar_chart(MODEL_LIST_FULL, 'marker_chart')
regard_bar_graph(
    models=MODEL_LIST_FULL, 
    img_name="all", 
    models_per_row=7
)