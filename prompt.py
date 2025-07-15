
from lib import *
from transformers import BertTokenizer, BertForMaskedLM
from transformers import logging
from evaluate import load
logging.set_verbosity_error()

perplexity_model = load("perplexity", module_type="metric")



def get_log_probability_for_word(sentence, target_word):
    model_name='bert-base-uncased'
    sentence = sentence + " "+ MASKBERT
    # Load model and tokenizer
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertForMaskedLM.from_pretrained(model_name)
    model.eval()

    inputs = tokenizer(sentence, return_tensors="pt")
    mask_index = (inputs['input_ids'][0] == tokenizer.mask_token_id).nonzero(as_tuple=True)[0].item()

    # Run through model
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits

    # Get probabilities for masked position
    probs = torch.nn.functional.softmax(logits[0, mask_index], dim=-1)
    log_probs = torch.log(probs)

    # Convert target word to token ID
    token_ids = tokenizer.encode(target_word, add_special_tokens=False)

    # # Check for multi-token words
    # if len(token_ids) != 1:
    #     print(f"⚠️ The word '{target_word}' was tokenized into {len(token_ids)} tokens: {tokenizer.convert_ids_to_tokens(token_ids)}")
    #     return None

    token_id = token_ids[0]
    log_prob = log_probs[token_id].item()
    return log_prob

def prompt_comparison(models):
    all_scores = [] 
    for model in models:
        if all(os.path.exists(f"{OUTPUT_SENTENCES}{prompt_num}/{model}.csv") for prompt_num in PROMPTS):
            for prompt_num in PROMPTS:
                df = pd.read_csv(f"{OUTPUT_SENTENCES}{prompt_num}/{model}.csv")
                df['Model'] = MODELS_LABELS.get(model, model)
                df['type'] = pd.Categorical(df['type'], categories=SUBJ_CATEGORIES, ordered=True)
                log_probabilities = []
                for _,row in tqdm(df.iterrows(), total= df.size, description= f"Calculating log probsbilities for {model} model with {prompt_num}"):
                    log_probabilities.append(get_log_probability_for_word(row.loc[UNMARKED], row.loc[PREDICTION]))
                display(df)
                df["log_probability"] = log_probabilities
                # predictions = {subj: df[df[TYPE] == subj][PREDICTION].dropna().values for subj in SUBJ_CATEGORIES}
                # log_probabilities = {subj: [] for subj in SUBJ_CATEGORIES}
                # for subj in SUBJ_CATEGORIES:
                #     for 
                #     log_probabilities[subj].append() = 
                    
                # perplexity = {subj: perplexity_model.compute(predictions=predictions[subj], model_id='gpt2')["mean_perplexity"] for subj in SUBJ_CATEGORIES}
                # display(predictions)
                # display(perplexity)

                # Perform ANOVA
                # for subj in SUBJ_CATEGORIES:
                #     perplexity.compute(predictions=predictions[subj], model_id='gpt2')

                    
                #print(f"{tool} - F-statistic: {f_statistic:.3f}, P-value: {p_value:.3g}")

                # # Save formatted result
                # result_row[f"{tool} F"] = round(f_statistic, 3)
                # result_row[f"{tool} p"] = round(p_value, 3)

            #all_scores.append(result_row)

    # Create final DataFrame
    # anova_results_df = pd.DataFrame(all_scores)
    # display(anova_results_df)
    # anova_results_df.to_csv(OUTPUT_TABLES + f'statistical_testing.csv', index=False)



#results = perplexity.compute(predictions=predictions, model_id='gpt2')

prompt_comparison(MODEL_LIST_FULL)