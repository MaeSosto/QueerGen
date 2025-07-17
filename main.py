from template import *
from model import *
from evaluation import *

# === Generate template ===
# template = Template()
# template.create_template()


# === Generate predictions ===
for model_name in MODEL_CLOSE:
    model = Model(model_name)
    for prompt_num in PROMPTS:
        model.get_predictions()

# === Evaluate predictions  ===
for model_name in MODEL_CLOSE:
    evaluation = Evaluation()
    for prompt_num in PROMPTS:
        evaluation.evaluate(model_name, prompt_num)
        



