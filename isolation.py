import pandas as pd
import re
import numpy as np
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
stopwords=set(nltk.corpus.stopwords.words('english'))
from collections import defaultdict
import joblib

#--------------Config variables----------------------
ISOLATION_SUBSET = None
SAMPLE_SIZE = 5000
CURRENT_MONTH = "june"
#----------------------------------------------------

data = joblib.load(f"outputs/data_{CURRENT_MONTH}_{SAMPLE_SIZE}.pkl")

data_cleaned = data[data['text'].apply(lambda x: bool(re.match('.*[a-zA-Z]+', x)))]

def extract_days(tokenized_sent, anchor_word='days', days_regex="[0-9.]+"):
    
    # find index position of anchor word in sentence
    anchor_pos = tokenized_sent.index(anchor_word)
    days=[]

    # search for the two words preceding the anchor word and check
    # if they are numbers (using days_regex to confirm). If so, add
    # them to `days` list and take average. This is useful if a range
    # has been given. e.g. 2-6 days will return 4 days.
    for i in [anchor_pos-2, anchor_pos-1]:
        if i >= 0 and bool(re.match(days_regex, tokenized_sent[i])):
            day = re.findall(days_regex, tokenized_sent[i])
            days.extend(day)

    if days == []:
        return None
    else:
        try:
            np.asarray(days, dtype=np.float32).mean()
        except:
            print(f"Days list {days} can't be converted to numpy array. Extracted from {tokenized_sent}")
        else:
            return np.asarray(days, dtype=np.float32).mean()


def recommended_isolation(data, keywords, anchor_word='days', subset=None, threshold=34):

    # take subset of data to avoid long run time if requested by user
    if subset is not None:
        data = data[:subset]

    isolation=[]
    index=[]
    keyword_counter = defaultdict(int)
    
    # loop through pandas dataframe
    for indx, row in data.iterrows():
        # split 'text' column into sentences
        sents = sent_tokenize(row['text'])
        # split those sentences into words
        words = [word_tokenize(sent) for sent in sents]
        # loop through list of lists where outer list is each sentence and
        # inner list is each word in that sentence. Convert to lower text
        # and remove stopwords. Then extract the no. of days from each sentence
        # that contains a keyword(s).
        for sent in words:
            sent_clean = [word.lower() for word in sent if word.lower() not in stopwords]
            keywords_present = [word for word in keywords if word in sent_clean]
            if len(keywords_present) > 0 and anchor_word in sent_clean:
                days_from_sent = extract_days(sent_clean)
                if days_from_sent is not None and days_from_sent < threshold:
                    isolation.append(days_from_sent)
                    index.append(indx)
                    for keyword_present in keywords_present:
                        keyword_counter[keyword_present] += 1
            
    return isolation, index, keyword_counter

keywords = ['quarantine', 'quarantining', 'quarantined', 'containment', 'notification', 'precautionary', 'quarantines', 'precautions', 'compulsory', 'tracing', 'precaution']
days_to_isolate = recommended_isolation(data_cleaned, keywords=keywords, subset=ISOLATION_SUBSET)

a = np.asarray(days_to_isolate[0], dtype=np.float32)
print(f'Median recommended isolation: {np.median(a)} days')
print(f'Number of keywords found: {len(days_to_isolate[0])}')

size_of_subset = SAMPLE_SIZE if ISOLATION_SUBSET is None else ISOLATION_SUBSET
joblib.dump(days_to_isolate, f"outputs/isolation_results_{CURRENT_MONTH}_{size_of_subset}.pkl")
print("Isolation results outputted to " + f"outputs/isolation_results_{CURRENT_MONTH}_{size_of_subset}.pkl")