from template import *
from model import *
from evaluation import *

# === Generate template ===
# template = Template()
# template.create_template()


# === Generate predictions ===
for model_name in [BERT_BASE, ROBERTA_BASE, LLAMA3, GEMMA3, GPT4_MINI]:
    model = Model(model_name)
    for prompt_num in PROMPTS:
        model.get_predictions(prompt_num)

# === Evaluate predictions  ===
for model_name in [BERT_BASE, ROBERTA_BASE, LLAMA3, GEMMA3, GPT4_MINI]:
    evaluation = Evaluation()
    for prompt_num in PROMPTS:
        evaluation.evaluate(model_name, prompt_num)