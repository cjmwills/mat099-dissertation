# import modules
import pandas as pd
import numpy as np
import re
import joblib
import glob
import json

#--------------Config variables----------------------
SAMPLE_SIZE = 5000
CURRENT_MONTH = "june"
PREVIOUS_MONTH = "april"
path_to_previous_data = "outputs/april_ids_5k.pkl"
path_to_current_data_glob = "data/2020-06/pdf_json/*"
#----------------------------------------------------


# import previous paper ID's
data_previous = joblib.load(path_to_previous_data)

# check there are no duplicates in the previous data
assert data_previous['paper_id'].nunique() == len(data_previous), "Duplicates in current paper ID's need to be removed"

# import current month data
# get paper ID's from pdf_json/ folder
all_papers_current = glob.glob(path_to_current_data_glob)
all_papers_current_id = [re.findall("([a-z0-9]+)\.json", paper)[0] for paper in all_papers_current]

# check there aren't any duplicate ID's in the current papers
assert len(set(all_papers_current_id)) == len(all_papers_current_id), "Duplicates in current paper ID's need to be removed"

# count the number of papers in both current and previous data
papers_in_both = data_previous['paper_id'].isin(all_papers_current_id).sum()

# filter for papers that were in both current and previous (to concatenate later with new papers)
previous_papers = data_previous[data_previous['paper_id'].isin(all_papers_current_id)]

# select the number of papers that ensures the SAMPLE_SIZE (5,000) papers are brought forward in total i.e. papers_in_both + X = SAMPLE_SIZE
new_papers = [paper for paper in all_papers_current_id if paper not in data_previous['paper_id'].values]
# convert to series to enable use of sample method
new_papers = pd.Series(new_papers, name='paper_id')
new_papers_sample = new_papers.sample(n=SAMPLE_SIZE-papers_in_both, random_state=42)

# sanity check that none of the new current papers were in the old previous papers
assert any(~new_papers_sample.isin(previous_papers)), "Some of the current paper sample are in the previous paper sample"

current_papers_to_use = pd.concat([previous_papers['paper_id'], new_papers_sample], ignore_index=True)

# confirm all the current papers to use are in the current data
assert all(True for paper in current_papers_to_use.values if paper in all_papers_current_id), "Some of the current paper sample are not in the raw data"

# import abstract and text body of each paper into pandas data frame.
data_json=[]
for paper in current_papers_to_use:
    paper_name = 'data/2020-06/pdf_json/' + paper + '.json'
    try:
        json.load(open(paper_name, 'rb'))
    except:
        pass
    else:
        paper_json = json.load(open(paper_name, 'rb'))
    data_json.append(paper_json)

# extract the abstracts and body text from each of the papers
abstracts=[]
body_texts=[]
for paper in data_json:
    abstract = paper['abstract']
    body_text = paper['body_text']
    # check if abstract is empty list
    if not abstract:
        abstracts.append(np.nan)
    else:
        abstracts.append(abstract[0]['text'])
    
    no_of_paragraphs = len(body_text)
    paragraphs=''
    for i in range(no_of_paragraphs):
        paragraph = body_text[i]['text']
        paragraphs += paragraph + '\n\n'
    body_texts.append(paragraphs)

assert len(abstracts) == len(body_texts) == SAMPLE_SIZE, f"The number of abstracts or body texts does not equal sample size ({SAMPLE_SIZE})"

data_dict = {'paper_id': current_papers_to_use.values, 'abstract': abstracts, 'text': body_texts}
data_current = pd.DataFrame(data_dict)

joblib.dump(data_current, f'outputs/data_{CURRENT_MONTH}_{SAMPLE_SIZE}.pkl')
print("Data outputted to " + f'outputs/data_{CURRENT_MONTH}_{SAMPLE_SIZE}.pkl')