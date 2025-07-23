from template import *
from model import *
from evaluation import *

# === Generate template ===
# template = Template()
# template.create_template()

# MODEL_LIST_FULL = [
#     BERT_BASE, BERT_LARGE, ROBERTA_BASE, ROBERTA_LARGE, LLAMA3,
#     LLAMA3_70B, GEMMA3, GEMMA3_27B, DEEPSEEK, DEEPSEEK_671B,
#     GPT4_MINI, GPT4, GEMINI_2_0_FLASH_LITE, GEMINI_2_0_FLASH
# ]

# === Generate predictions ===
# for model_name in [DEEPSEEK_671B]:
#     model = Model(model_name)
#     #for prompt_num, _ in enumerate(PROMPTS):
#     proc = model.get_predictions(1)
#         # if proc is None:
#         #     break

# === Evaluate predictions  ===
for model_name in MODEL_LIST_FULL :
    evaluation = Evaluation()
    for prompt_num, _ in enumerate(PROMPTS):
        if not os.path.exists(f"{OUTPUT_SENTENCES}prompt_{prompt_num}/{model_name}.csv"):
                continue
        evaluation.evaluate(model_name, prompt_num)