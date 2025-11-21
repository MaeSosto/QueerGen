from src.template import *
from src.model import *
from src.evaluation import *

NUM_PREDICTION = 5
# === Generate template ===
template = Template()
template.create_template()
error = False

sample_from_dataset()

# # === Generate predictions ===
logger.info("🧬 Generate Predictions 🧬")
for model_name in MODEL_LIST_FULL:
    model = Model(model_name, NUM_PREDICTION)
    if error:
        break        
    error = model.get_predictions(PROMPT_DEFAULT)
    if error:
        break

if not error:
    logger.info("📊 Evaluate Predictions 📊")
    # === Evaluate predictions  ===
    for model_name in MODEL_LIST_FULL:
        evaluation = Evaluation()
        if not os.path.exists(f"{PATH_GENERATIONS}prompt_{PROMPT_DEFAULT}/{model_name}.csv"):
            continue
        error = evaluation.evaluate(model_name, PROMPT_DEFAULT)
        if error:
            break
        