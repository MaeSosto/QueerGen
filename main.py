from src.template import *
from src.model import *
from src.evaluation import *

NUM_PREDICTION = 5

# === Generate template ===
template = Template()
template.create_template()

#Sample 300 items if more than 1
if NUM_PREDICTION > 1:
    sample_from_dataset() 
    
error = False

#=== Generate predictions ===
logger.info("🧬 Generate Predictions 🧬")
for model_name in [
DEEPSEEK_R1_671B,
    GPT4O_MINI, GPT4O, GEMINI_2_0_FLASH_LITE, GEMINI_2_0_FLASH
]:
    model = Model(model_name, NUM_PREDICTION)
    if error:
        break
    if NUM_PREDICTION == 1:
        for prompt_num, _ in enumerate(PROMPTS):
            error = model.get_predictions(prompt_num)
            if error:
                break
        if error:
                break
    else:
        error = model.get_predictions(PROMPT_DEFAULT)
        if error:
            break

if not error:
    logger.info("📊 Evaluate Predictions 📊")
    # === Evaluate predictions  ===
    for model_name in MODEL_LIST_FULL:
        evaluation = Evaluation(model_name, NUM_PREDICTION)
        if NUM_PREDICTION == 1:
            for prompt_num, _ in enumerate(PROMPTS):
                error = evaluation.evaluate(prompt_num)
                if error:
                    break
            if error:
                break
        else:
            error = evaluation.evaluate(PROMPT_DEFAULT)
            if error:
                break

    