import argparse
import os
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument('--results_dir', type=str, default='../data/korbit_ref_results')
args = parser.parse_args()

edu_list = []
scores = []

for file in os.listdir(args.results_dir):
    with open(os.path.join(args.results_dir, file), 'r') as f:
        file_contents = []
        for line in f:
            file_contents.append(line.strip())

        scores.append(float(file_contents[-1]))
        edu_list.append(file_contents[:-1])

data = {'edus': edu_list, 'scores': scores}
edu_df = pd.DataFrame.from_dict(data)
edu_df.sort_values('scores', inplace=True).reset_index(drop=True)
edu_df.to_pickle(args.results_dir.split("/")[-1] + ".pk")
