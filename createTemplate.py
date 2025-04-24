from lib.constants import *
TEMPLATE_PATH = DATA_SOURCE + 'templates.csv'
SUBJECT_PATH = DATA_SOURCE + 'subjects.csv'
MARKER_PATH = DATA_SOURCE + 'markers.csv'
TEMPLATE_PATH_COMPLETE = DATA_SOURCE + 'template_complete.csv'

def createTemplate():
    templateFile = pd.read_csv(TEMPLATE_PATH)
    dataList =[]
    logger.info("๏ Generating template...")
    
    for _,row in tqdm(templateFile.iterrows(), total=templateFile.shape[0], desc='Creating template', unit=' s'):
        subjectFile = pd.read_csv(SUBJECT_PATH)
        markerFile = pd.read_csv(MARKER_PATH)
        
        sentence = row.loc[TEMPLATE]
        #Creating sentences with nouns
        for _, subjRow in subjectFile.iterrows():
            subject = subjRow.loc[VALUE] 
            dataList.append([
                sentence, #original
                subject, #subject
                UNMARKED,
                subjRow.loc[TYPE], #type
                "neutral", #category
                sentence.replace(SUBJECT_, f"The {subject}"), #unmarked
                sentence.replace(SUBJECT_, f"The {subject}") #marked
            ])    
            for _,markerRow in markerFile.iterrows():
                marker = markerRow.loc[VALUE]
                sub = f"The {marker} {subject}" if markerRow.loc[ADJ] == "y" else f"The {subject} is a {marker} and"
                sub = sentence.replace(SUBJECT_, sub)
                
                dataList.append([
                    sentence, #original
                    subject, #subject
                    marker, #marker 
                    markerRow.loc[TYPE], #type
                    markerRow.loc[CATEGORY], #category
                    sentence.replace(SUBJECT_, f"The {subject}"), #unmarked
                    sub #marked
                ]) 
    data_df = pd.DataFrame(dataList, columns=[TEMPLATE, SUBJECT, MARKER, TYPE, CATEGORY, UNMARKED, MARKED])
    data_df.to_csv(TEMPLATE_PATH_COMPLETE, index_label = 'index')
    logger.info("๏ File template generated!")

createTemplate()
