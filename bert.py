from transformers import BertTokenizer, TFBertForQuestionAnswering
import tensorflow as tf
import matplotlib.pyplot as plt
from collections import Counter
import pandas as pd
import numpy as np
import joblib
from tqdm import tqdm

#--------------Config variables----------------------
BERT_SAMPLE_SIZE = 500
SAMPLE_SIZE = 5000
CURRENT_MONTH = "august"
QUESTION = "What are the risk factors of Covid-19?"
#----------------------------------------------------

tokenizer = BertTokenizer.from_pretrained("dmis-lab/biobert-base-cased-v1.1-squad")
model = TFBertForQuestionAnswering.from_pretrained("dmis-lab/biobert-base-cased-v1.1-squad", from_pt=True)

data = joblib.load(f"outputs/data_{CURRENT_MONTH}_{SAMPLE_SIZE}.pkl")

not_null_data = data[data['abstract'].notnull()]

def get_answer(question, text):
    
    inputs = tokenizer(question, text, return_tensors="tf")
    outputs = model(**inputs)
    answer_start_index = int(tf.math.argmax(outputs.start_logits, axis=-1)[0])
    answer_end_index = int(tf.math.argmax(outputs.end_logits, axis=-1)[0])
    predict_answer_tokens = inputs.input_ids[0, answer_start_index : answer_end_index + 1]
    predicted_answer = tokenizer.decode(predict_answer_tokens)

    return predicted_answer

qa_results=[]
for index, row in tqdm(not_null_data[:10].iterrows(), total=not_null_data[:10].shape[0]):

    if len(tokenizer.tokenize(row['abstract'])) < 475:
        
        text = row['abstract']
        predicted_answer = get_answer(question=QUESTION, text=text)
        
        if predicted_answer != '[CLS]' and predicted_answer != '':
            qa_results.append((index, predicted_answer))

joblib.dump(qa_results, f'outputs/bert_qa_results_{CURRENT_MONTH}_{BERT_SAMPLE_SIZE}.pkl')

print("BERT QA results outputted to " + f'outputs/bert_qa_results_{CURRENT_MONTH}_{BERT_SAMPLE_SIZE}.pkl')

# updated risk factors from word2vec
age_synonyms = ['age', 'younger', 'older', 'ages', 'aged', 'old']
gender_synonyms = ['gender', 'ethnicity', 'sex', 'demographics', 'occupation', 'nationality']
sex_synonyms = ['sex', 'gender', 'ethnicity', 'breed', 'race/ethnicity', 'occupation']
pneumonia_synonyms = ['pneumonia', 'pneumonias', 'tracheitis', 'croup', 'pharyngitis', 'bronchiolitis']
obesity_synonyms = ['obesity', 'diabetes', 'hypertension', 'malnutrition', 'osteoporosis', 'nafld']
diabetes_synonyms = ['diabetes', 'mellitus', 'obesity', 'hypertension', 'insulin-dependent', 'hypercholesterolemia']
smoking_synonyms = ['smoking', 'cigarette', 'smoke', 'obesity', 'abuse', 'breastfeeding']
cardiovascular_synonyms = ['cardiovascular', 'cardiopulmonary', 'cardiac', 'cerebrovascular', 'hypertension', 'musculoskeletal']
location_synonyms = ['location', 'locations', 'geographical', 'distribution', 'geographic', 'temporal']
contact_synonyms = ['contact', 'contacts', 'transmission', 'proximity', 'movement', 'exposure']
asthma_synonyms = ['asthma', 'wheezing', 'copd', 'exacerbations', 'atopy', 'wheeze']
cancer_synonyms = ['cancer', 'cancers', 'carcinoma', 'prostate', 'melanoma', 'tumour']

candidate_risk_factors = set(age_synonyms + gender_synonyms + sex_synonyms + pneumonia_synonyms + obesity_synonyms + diabetes_synonyms + smoking_synonyms + cardiovascular_synonyms + location_synonyms + contact_synonyms + asthma_synonyms + cancer_synonyms)

risk_factor_list=[]
for result in qa_results:
    for risk_factor in candidate_risk_factors:
        if risk_factor in result[1].split():
            risk_factor_list.append(risk_factor)

risk_factor_count = Counter(risk_factor_list)

risk_factor_df = pd.DataFrame.from_dict(risk_factor_count, orient='index')
risk_factor_df.columns = ['count']
risk_factor_df.sort_values(by='count', ascending=True).plot(kind='barh', legend=None)
plt.savefig(f"outputs/plots/biobert_risk_factors_long_{CURRENT_MONTH}")

print("BERT QA plot outputted to " + f"outputs/plots/biobert_risk_factors_long_{CURRENT_MONTH}")