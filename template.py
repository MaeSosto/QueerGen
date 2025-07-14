from lib import *

TEMPLATE_PATH = DATA_SOURCE + 'templates.csv'
SUBJECT_PATH = DATA_SOURCE + 'subjects.csv'
MARKER_PATH = DATA_SOURCE + 'markers.csv'
TEMPLATE_PATH_COMPLETE = DATA_SOURCE + 'template_complete.csv'
ADJ = 'adj'

class Template:
    
    def __init__(self):
        self.templates = pd.read_csv(TEMPLATE_PATH)
        self.subjects = pd.read_csv(SUBJECT_PATH)
        self.markers = pd.read_csv(MARKER_PATH)
        
        
    def create_template(self):
        #logger.info("๏ Loading input files...")
        data_list = []
        logger.info("๏ Generating template sentences...")

        for _, template_row in tqdm(self.templates.iterrows(), total=self.templates.shape[0], desc='Creating template', unit=' s'):
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
        logger.info("๏ File template generated!")
        
    