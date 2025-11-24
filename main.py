from src.template import *
from src.model import *
from src.evaluation import *
from src.graphs import *

#CHANGE THE FOLLOWING FLAG ACCORDING ON THE REQUIRED NUMBER OF PREDICTIONS
NUM_PREDICTION = 1

# === Generate template ===
template = Template()
template.create_template()

#Sample 300 items (100 of each subj cat) if more than 1 prediction
if NUM_PREDICTION > 1:
    sample_from_dataset() 
    
error = False

#=== Generate predictions ===
logger.info("🧬 Generate Predictions 🧬")
for model_name in MODEL_LIST_FULL:
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
    # === Evaluate predictions  ===
    logger.info("📊 Evaluate Predictions 📊")
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

graphs(NUM_PREDICTION)