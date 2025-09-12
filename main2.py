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
DEEPSEEK_671B
]:
    model = Model(model_name)
    initialized = False
    if error:
        break
    for prompt_num, _ in enumerate(PROMPTS):
        if not initialized:
            error = model.initialize_model()
            initialized = True
        error = model.get_predictions(prompt_num)
        if error:
            break
    if error:
            break

if not error:
    logger.info("📊 Evaluate Predictions 📊")
    # === Evaluate predictions  ===
    for model_name in [
DEEPSEEK_671B
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
    