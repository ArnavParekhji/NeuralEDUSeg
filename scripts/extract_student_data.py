import pandas as pd
import ast
import unicodedata
from nltk import word_tokenize
import re, os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--dump_dir', type=str, default='../data/rst/student_data')
args = parser.parse_args()

all_data_df = pd.read_csv('student_exercise_dataset.tsv', sep='\t', header=0)
utterance_raw_series = all_data_df['list_of_user_utterances']

utterance_series = utterance_raw_series.map(lambda x : ast.literal_eval(x)) # Turn raw string into Python object
utterance_series = utterance_series[~(utterance_series.str.len() == 0)]  # Remove empty lists
utterance_series = utterance_series.explode() # Explode lists into new rows
utterance_series = utterance_series.map(lambda x : unicodedata.normalize('NFKD', x).replace("  ", " ").strip())  # Remove bad encoding
utterance_series = utterance_series.map(lambda x : re.sub("[\s]+", " ", x).strip()) # Remove extra whitespace characters
utterance_series = utterance_series[~utterance_series.str.contains("\$.*\$")] # Remove formulas
utterance_series = utterance_series[utterance_series.map(lambda x : len(word_tokenize(x)) >= 5)] # Remove answers with less than 5 words

for idx, utterance in utterance_series.iteritems():
    with open(os.path.join(args.dump_dir, "student_resp_{}.txt".format(idx)), 'w') as f:
        f.write(utterance)
