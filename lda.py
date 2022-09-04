import pandas as pd
from tqdm import tqdm
tqdm.pandas()

import re
import numpy as np
from collections import defaultdict
import joblib
import matplotlib.pyplot as plt

import nltk
stopwords=set(nltk.corpus.stopwords.words('english'))
from langdetect import detect

import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation

#--------------Config variables----------------------
SAMPLE_SIZE = 5000
CURRENT_MONTH = "august"
#----------------------------------------------------

data = joblib.load(f"outputs/data_{CURRENT_MONTH}_{SAMPLE_SIZE}.pkl")

# only keep rows that contain an alpha character (a letter) as detect() throws an error if this isn't the case
data_cleaned = data[data['text'].apply(lambda x: bool(re.match('.*[a-zA-Z]+', x)))]
data_cleaned.reset_index(inplace=True, drop=True)

lang = data_cleaned['text'].progress_apply(detect)
joblib.dump(lang, f"outputs/lda_lang_{CURRENT_MONTH}_{SAMPLE_SIZE}.pkl")
print("Language detector outputted to " + f"outputs/lda_lang_{CURRENT_MONTH}_{SAMPLE_SIZE}.pkl")

data_eng = data_cleaned[lang == 'en']
print('Rows before removing non-english:', data_cleaned.shape[0])
print('Rows after removing non-english:', data_eng.shape[0])

stopwords=set(nltk.corpus.stopwords.words('english'))

vectorizer = TfidfVectorizer(strip_accents='ascii', stop_words=stopwords)
vectorizer.fit(data_eng['text'])

joblib.dump(vectorizer, f"outputs/lda_vectorizer_{CURRENT_MONTH}_{SAMPLE_SIZE}.pkl")
print("LDA vectorizer outputted to " + f"outputs/lda_vectorizer_{CURRENT_MONTH}_{SAMPLE_SIZE}.pkl")

data_vectorized = vectorizer.fit_transform(data_eng['text'])

word_count = pd.DataFrame({'word': vectorizer.get_feature_names_out(), 'count': np.asarray(data_vectorized.sum(axis=0))[0]})
word_count.sort_values('count', ascending=False).set_index('word')[:20].sort_values('count', ascending=True).plot(kind='barh', xlabel='', legend=None, title=f"TF-IDF by word, {CURRENT_MONTH.capitalize()}")
plt.savefig(f"outputs/plots/lda_vectorizer_freq_{CURRENT_MONTH}")

lda = LatentDirichletAllocation(n_components=5)
lda.fit(data_vectorized)

joblib.dump(lda, f"outputs/lda_model_{CURRENT_MONTH}_{SAMPLE_SIZE}.pkl")
print("LDA model outputted to " + f"outputs/lda_model_{CURRENT_MONTH}_{SAMPLE_SIZE}.pkl")