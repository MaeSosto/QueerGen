from template import *
from model import *
from evaluation import *

# === Generate template ===
#template = Template()
#template.create_template()


# === Generate predictions ===
for model_name in [BERT_BASE]:
    model = Model(model_name)
    for prompt_num in PROMPTS:
        model.get_predictions(prompt_num)

# === Evaluate predictions  ===
for model_name in [GPT4_MINI]:
    evaluation = Evaluation()
    for prompt_num in PROMPTS:
        evaluation.evaluate(model_name, prompt_num)
        



