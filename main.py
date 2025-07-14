from template import *
from model import *
from evaluation import *

#template = Template()
#template.create_template()


# === Execution ===
for prompt_num in PROMPTS:
    for model_name in [LLAMA3, GPT4_MINI]:
        model = Model(model_name)
        model.get_predictions(prompt_num)

# for model_name in [LLAMA3, GPT4_MINI]:
#         evaluation = Evaluation()
#         evaluation.evaluate(model_name, PROMPTS[0])
        



