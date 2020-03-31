import pandas as pd
import numpy as np
import unicodedata
import ast, re, os
import argparse
from tqdm import tqdm

import pickle
import logging
import spacy
from rst_edu_reader import RSTData
from atten_seg import AttnSegModel
from seg_config import parse_args

from nltk import word_tokenize
import tensorflow as tf
from sklearn.metrics.pairwise import cosine_similarity as cos_sim
from sentence_transformers import SentenceTransformer


def preprocess_ref_data(args):
    # Load reference data into DataFrame
    ref_df = pd.read_csv(args.ref_data_file, sep='\t')
    ref_df.drop(columns='Unnamed: 0', inplace=True)  # Redundant column

    # Clean up data: transform strings to lists, remove equations, less than 5 words, fix whitespaces
    for col in ['problem_texts', 'reference_solutions', 'reference_nonsolutions']:
        ref_df[col] = ref_df[col].map(ast.literal_eval)

    # Problem texts are in singleton lists, map them into strings
    assert not (ref_df['problem_texts'].str.len() > 1).any()
    ref_df['problem_texts'] = ref_df['problem_texts'].map(lambda x : x[0])

    # Explode the reference solutions and nonsolutions
    sol_df = ref_df[['exercise_index', 'problem_texts', 'reference_solutions']].explode('reference_solutions').reset_index(drop=True)
    nonsol_df = ref_df[['exercise_index', 'problem_texts', 'reference_nonsolutions']].explode('reference_nonsolutions').dropna().reset_index(drop=True)

    # Remove equations
    sol_df = sol_df[~sol_df.reference_solutions.str.contains("\$.*\$")].reset_index(drop=True)
    nonsol_df = nonsol_df[~nonsol_df.reference_nonsolutions.str.contains("\$.*\$")].reset_index(drop=True)

    # Remove rows with less than 5 words
    sol_df = sol_df[sol_df.reference_solutions.apply(lambda x : len(word_tokenize(x))) >= 5].reset_index(drop=True)
    nonsol_df = nonsol_df[nonsol_df.reference_nonsolutions.apply(lambda x : len(word_tokenize(x))) >= 5].reset_index(drop=True)

    return sol_df, nonsol_df


def preprocess_std_data(args):
    std_df = pd.read_csv(args.std_data_file, sep='\t')
    drop_cols = ['Unnamed: 0', 'user_id', 'course_id', 'multiple_choice_question', 'exercise_shown_time', 'total_time_spent', 'list_of_user_utterances_probs']
    std_df.drop(columns=drop_cols, inplace=True)
    std_df['list_of_user_utterances'] = std_df['list_of_user_utterances'].map(ast.literal_eval) # Problem texts are literal strings, map them into lists of strings
    std_df = std_df.explode('list_of_user_utterances').dropna().reset_index(drop=True) # Explode the student's attempts into individual answers
    std_df = std_df[~std_df.list_of_user_utterances.str.contains("\$.*\$")].reset_index(drop=True)  # Remove equations
    std_df = std_df[std_df.list_of_user_utterances.apply(lambda x : len(word_tokenize(x))) >= 5].reset_index(drop=True)

    std_df.rename(columns={'list_of_user_utterances': 'std_response'}, inplace=True)
    std_df['std_response'] = std_df['std_response'].map(lambda x : unicodedata.normalize('NFKD', x))  # Remove bad encodings
    std_df['std_response'] = std_df['std_response'].map(lambda x : re.sub("[\s]+", " ", x).strip())  # Remove extra spaces
    return std_df


def preprocess_pdtb_data(args):
    raw_df = pd.read_json(os.path.join(args.pdtb_json_dir, "{}_raw.json".format(args.pdtb_dataset)), typ='series')
    return raw_df.to_frame(name='pdtb_text')


def segment_data(dfs, col_names):
    """Segment the given dataframes into EDUs, add the EDUs into the dataframes and return"""
    args = parse_args()
    np.random.seed(args.seed)
    tf.set_random_seed(args.seed)

    # Logging
    logger = logging.getLogger("SegEDU")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # Loading
    rst_data = RSTData()
    logger.info('Loading vocab...')
    with open(args.word_vocab_path, 'rb') as fin:
        word_vocab = pickle.load(fin)
        logger.info('Word vocab size: {}'.format(word_vocab.size()))
    rst_data.word_vocab = word_vocab
    logger.info('Loading the model...')
    model = AttnSegModel(args, word_vocab)
    model.restore('best', args.model_dir)
    if model.use_ema:
        model.sess.run(model.ema_backup_op)
        model.sess.run(model.ema_assign_op)

    spacy_nlp = spacy.load('en', disable=['parser', 'ner', 'textcat'])

    for df, col_name in zip(dfs, col_names):
        edu_results = {}
        for idx, row in tqdm(df.iterrows(), total=len(df.index)):
            try:
                # logger.info('Segmenting example {}...'.format(idx))
                raw_sents = [row[col_name]]
                samples = []
                for sent in spacy_nlp.pipe(raw_sents, batch_size=1000, n_threads=5):
                    samples.append({'words': [token.text for token in sent],
                                    'words_ws': [token.text_with_ws for token in sent],
                                    'edu_seg_indices': []})
                rst_data.test_samples = samples
                data_batches = rst_data.gen_mini_batches(args.batch_size, test=True, shuffle=False)

                edus = []
                for batch in data_batches:
                    batch_pred_segs = model.segment(batch)
                    for sample, pred_segs in zip(batch['raw_data'], batch_pred_segs):
                        one_edu_words = []
                        for word_idx, word in enumerate(sample['words_ws']):
                            if word_idx in pred_segs:
                                edus.append(''.join(one_edu_words))
                                one_edu_words = []
                            one_edu_words.append(word)
                        if one_edu_words:
                            edus.append(''.join(one_edu_words))

                edu_results[idx] = edus
            except:
                logger.info("Crashed while segmenting {}.".format(idx))
                edu_results[idx] = []
                continue

        df['edus'] = pd.Series(edu_results)
    merged = pd.concat(dfs).reset_index(drop=True)
    merged = merged[merged['edus'].map(lambda x: len(x)) > 0]  # Remove rows with unsegmentable EDUs
    return merged


def semantic_equivalence_embeds(data_df):
    semeq_model = SentenceTransformer('roberta-base-nli-stsb-mean-tokens')
    data_df['semeq_embedding'] = data_df['edus'].map(semeq_model.encode)
    return

if __name__ == "__main__":
    args = parse_args()

    if args.dataset == 'ref':
        sol_df, nonsol_df = preprocess_ref_data(args)
        edu_df = segment_data([sol_df, nonsol_df], ['reference_solutions', 'reference_nonsolutions'])
        semantic_equivalence_embeds(edu_df)
        pd.to_pickle(edu_df, "ref_encoded_edus.pk")
    elif args.dataset == 'student':
        std_df = preprocess_std_data(args)
        edu_df = segment_data([std_df], ['std_response'])
        semantic_equivalence_embeds(edu_df)
        pd.to_pickle(edu_df, "std_encoded_edus.pk")
    elif args.dataset == 'pdtb':
        pdtb_df = preprocess_pdtb_data(args)
        edu_df = segment_data([pdtb_df], ['pdtb_text'])
        pd.to_pickle(pdtb_df, "{}_edus.pk".format(args.pdtb_dataset))
    else:
        raise ValueError("Invalid dataset choice: {}".format(args.dataset))
