# -*- coding: utf-8 -*- #
"""*********************************************************************************************"""
#   FileName     [ generate_len_for_bucket.py ]
#   Synopsis     [ preprocess audio speech to generate meta data for dataloader bucketing ]
#   Author       [ Andy T. Liu (Andi611) ]
#   Copyright    [ Copyleft(c), Speech Lab, NTU, Taiwan ]
#   Reference    [ https://github.com/Alexander-H-Liu/End-to-end-ASR-Pytorch ]
"""*********************************************************************************************"""


###############
# IMPORTATION #
###############
import os
import argparse
import torchaudio
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from joblib import Parallel, delayed
from datasets import load_dataset

os.environ["MKL_DISABLE_FAST_MM"] = "1"

#############################
# PREPROCESS CONFIGURATIONS #
#############################
def get_preprocess_args():
    
    parser = argparse.ArgumentParser(description='preprocess arguments for any dataset.')

    # parser.add_argument('-i', '--input_data', default='../LibriSpeech/', type=str, help='Path to your LibriSpeech directory', required=False)
    parser.add_argument('-o', '--output_path', default='./data/ted', type=str, help='Path to store output', required=False)
    # parser.add_argument('-a', '--audio_extension', default='.flac', type=str, help='audio file type (.wav / .flac / .mp3 / etc)', required=False)
    parser.add_argument('-n', '--name', default='len_for_bucket', type=str, help='Name of the output directory', required=False)
    # parser.add_argument('--n_jobs', default=-1, type=int, help='Number of jobs used for feature extraction', required=False)

    args = parser.parse_args()
    return args


##################
# EXTRACT LENGTH #
##################
def extract_length(instance):
    return len(instance["audio"]["array"])


###################
# GENERATE LENGTH AND FILTER OUT 10HR HIGH QUALITY SUBSET #
###################
def generate_length(args, tr_set):
    #ds = load_dataset("LIUM/tedlium", "release3", trust_remote_code=True)
    ds = load_dataset("LIUM/tedlium", "release3")
    for i, s in enumerate(tr_set):
        if s == "train":
            max_sec = 3600 * 10
        else:
            max_sec = 2e9
        split_ds = ds[s]
        sr = split_ds.features["audio"].sampling_rate

        print('')
        print(f'Preprocessing data in: {s}, {len(split_ds)} audio files found.')

        output_dir = os.path.join(args.output_path, args.name)
        if not os.path.exists(output_dir): os.makedirs(output_dir)

        print('Extracting audio length...', flush=True)
        total = 0
        todo, tr_x = [], []
        for idx, instance in tqdm(enumerate(split_ds)):
            duration = extract_length(instance)
            # Filter out too short/long
            if duration / sr < 1 or duration / sr > 60:
                continue

            # Filter out <unk>
            if "<unk>" in instance["text"]:
                continue
            if len(todo) < 5:
                print(instance["text"])
                
            todo.append(idx)
            tr_x.append(duration)
            total += duration
            if total > max_sec * sr:  # for training, only use 10hr
                print("max_sec achieved.")
                break

        # sort by len
        sorted_todo = [todo[idx] for idx in reversed(np.argsort(tr_x))]
        # Dump data
        df = pd.DataFrame(data={'file_path':sorted_todo, 'length':list(reversed(sorted(tr_x))), 'label':None})
        df.to_csv(os.path.join(output_dir, tr_set[i] + '.csv'))

    print('All done, saved at', output_dir, 'exit.')


########
# MAIN #
########
def main():

    # get arguments
    args = get_preprocess_args()
    tr_set = ["train", "validation", "test"]  # ted
    generate_length(args, tr_set)


if __name__ == '__main__':
    main()
