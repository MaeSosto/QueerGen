from src.template import *
from src.model import *
from src.evaluation import *

# === Generate template ===
template = Template()
template.create_template()

# # === Generate predictions ===
logger.info("🧬 Generate Predictions 🧬")
error = False
for model_name in [
    BERT_BASE, BERT_LARGE, ROBERTA_BASE, ROBERTA_LARGE, LLAMA3,
    LLAMA3_70B, GEMMA3, GEMMA3_27B, DEEPSEEK_671B,
    GPT4_MINI, GPT4, GEMINI_2_0_FLASH_LITE, GEMINI_2_0_FLASH
]:
    model = Model(model_name)
    for prompt_num, _ in enumerate(PROMPTS):
        error = model.get_predictions(prompt_num)
        if error:
            break
    if error:
            break

logger.info("📊 Evaluate Predictions 📊")
# === Evaluate predictions  ===
for model_name in [
    BERT_BASE, BERT_LARGE, ROBERTA_BASE, ROBERTA_LARGE, LLAMA3,
    LLAMA3_70B, GEMMA3, GEMMA3_27B, DEEPSEEK_671B,
    GPT4_MINI, GPT4, GEMINI_2_0_FLASH_LITE, GEMINI_2_0_FLASH
]:
    evaluation = Evaluation()
    for prompt_num, _ in enumerate(PROMPTS):
        if not os.path.exists(f"{PATH_GENERATIONS}prompt_{prompt_num}/{model_name}.csv"):
            continue
        error = evaluation.evaluate(model_name, prompt_num)
        if error:
            break
    if error:
        break
    