import ast
import pandas as pd
from nltk import word_tokenize
import os

base_data_dir = '../data/rst/korbit_full/'
data_file = 'preprocessed_reference_exercise_data.tsv'
result_dir = '../data/korbit_ref_results/'

ref_df = pd.read_csv(os.path.join(base_data_dir, data_file), sep='\t')

for col in ['problem_texts', 'reference_solutions', 'reference_nonsolutions']:
	ref_df[col] = ref_df[col].map(ast.literal_eval)

# Make sure all lists have length 1
assert not (ref_df['problem_texts'].str.len() > 1).any()
ref_df['problem_texts'] = ref_df['problem_texts'].map(lambda x : x[0])

sol_df = ref_df[['exercise_index', 'reference_solutions']].explode('reference_solutions').reset_index(drop=True)
nonsol_df = ref_df[['exercise_index', 'reference_nonsolutions']].explode('reference_nonsolutions').dropna().reset_index(drop=True)

# Filter out all cells with equations.
ref_df = ref_df[~ref_df.problem_texts.str.contains("\$.*\$")].reset_index(drop=True)
sol_df = sol_df[~sol_df.reference_solutions.str.contains("\$.*\$")].reset_index(drop=True)
nonsol_df = nonsol_df[~nonsol_df.reference_nonsolutions.str.contains("\$.*\$")].reset_index(drop=True)

# Filter out rows with less than 5 words
ref_df = ref_df[ref_df.problem_texts.apply(lambda x : len(word_tokenize(x))) >= 5].reset_index(drop=True)
sol_df = sol_df[sol_df.reference_solutions.apply(lambda x : len(word_tokenize(x))) >= 5].reset_index(drop=True)
nonsol_df = nonsol_df[nonsol_df.reference_nonsolutions.apply(lambda x : len(word_tokenize(x))) >= 5].reset_index(drop=True)

# Write all to file
for idx, content in ref_df['problem_texts'].iteritems():
	with open(os.path.join(result_dir, "problem_{}.txt".format(idx)), 'w') as f:
		f.write(content)

for idx, content in sol_df['reference_solutions'].iteritems():
	with open(os.path.join(result_dir, "solution_{}.txt".format(idx)), 'w') as f:
		f.write(content)

for idx, content in nonsol_df['reference_nonsolutions'].iteritems():
	with open(os.path.join(result_dir, "nonsolution_{}.txt".format(idx)), 'w') as f:
		f.write(content)

