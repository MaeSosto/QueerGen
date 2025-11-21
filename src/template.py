from src.lib import *
ADJ = 'adj'

class Template:
    
    def __init__(self):
        self.templates = pd.read_csv(TEMPLATE_PATH)
        self.subjects = pd.read_csv(SUBJECT_PATH)
        self.markers = pd.read_csv(MARKER_PATH)
        
        
    def create_template(self):
        if not os.path.exists(TEMPLATE_PATH_COMPLETE):
            data_list = []
            for _, template_row in tqdm(self.templates.iterrows(), total=self.templates.shape[0], desc="✍️ Generating template", unit=' s'):
                template = template_row[TEMPLATE]

                for _, subject_row in self.subjects.iterrows():
                    subject = subject_row[VALUE]
                    unmarked_sentence = re.sub(SUBJECT_, subject, template)

                    # Add unmarked example
                    data_list.append([
                        template,
                        subject,
                        UNMARKED,
                        subject_row[TYPE],
                        "neutral",
                        unmarked_sentence,
                        unmarked_sentence
                    ])

                    for _, marker_row in self.markers.iterrows():
                        marker = marker_row[VALUE]
                        is_adj = marker_row[ADJ]

                        if is_adj == "y":
                            marked_sentence = re.sub(SUBJECT_, f"{marker} {subject}", template)
                        else:
                            base_sentence = re.sub(SUBJECT_, subject, template)
                            marked_sentence = re.sub(SUBJECT_, f"{subject} is a {marker} and", template)

                        data_list.append([
                            template,
                            subject,
                            marker,
                            marker_row[TYPE],
                            marker_row[CATEGORY],
                            unmarked_sentence,
                            marked_sentence
                        ])

            df = pd.DataFrame(data_list, columns=[TEMPLATE, SUBJECT, MARKER, TYPE, CATEGORY, UNMARKED, MARKED])
            df.to_csv(TEMPLATE_PATH_COMPLETE, index_label='index')
        logger.info("📝 Template ready!")

def sample_from_dataset():
    import random
    import pandas as pd

    # Load dataset
    df = pd.read_csv(TEMPLATE_PATH_COMPLETE)

    # Filter by type
    df_unmarked = df[df[TYPE] == UNMARKED]
    df_queer = df[df[TYPE] == QUEER]
    df_nonqueer = df[df[TYPE] == NONQUEER]

    # ---- Ensure UNMARKED baseline is 100 ----
    if len(df_unmarked) != 100:
        raise ValueError(f"UNMARKED baseline must be exactly 100 rows (found {len(df_unmarked)}).")

    # For each UNMARKED, sample 1 QUEER and 1 NONQUEER with same TEMPLATE+SUBJECT
    sampled_queer_rows = []
    sampled_nonqueer_rows = []

    for _, row in df_unmarked.iterrows():
        template = row["template"]
        subject = row["subject"]

        # Filter QUEER and NONQUEER with same template+subject
        queer_group = df_queer[
            (df_queer["template"] == template) &
            (df_queer["subject"] == subject)
        ]
        nonqueer_group = df_nonqueer[
            (df_nonqueer["template"] == template) &
            (df_nonqueer["subject"] == subject)
        ]

        # Sample one randomly
        sampled_queer_rows.append(queer_group.sample(n=1, random_state=random.randint(0, 10000)))
        sampled_nonqueer_rows.append(nonqueer_group.sample(n=1, random_state=random.randint(0, 10000)))

    # Concatenate all sampled rows
    sample_queer = pd.concat(sampled_queer_rows, ignore_index=True)
    sample_nonqueer = pd.concat(sampled_nonqueer_rows, ignore_index=True)

    # Final dataset
    df_final = pd.concat([df_unmarked, sample_queer, sample_nonqueer], ignore_index=True)

    # Save
    df_final.to_csv(TEMPLATE_PATH_TOP5, index=False)