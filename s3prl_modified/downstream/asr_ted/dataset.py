# -*- coding: utf-8 -*- #
"""*********************************************************************************************"""
#   FileName     [ dataset.py ]
#   Synopsis     [ the phone dataset ]
#   Author       [ S3PRL, Xuankai Chang ]
#   Copyright    [ Copyleft(c), Speech Lab, NTU, Taiwan ]
"""*********************************************************************************************"""


###############
# IMPORTATION #
###############
import logging
import os
import random
import re
#-------------#
import pandas as pd
from tqdm import tqdm
from pathlib import Path
#-------------#
import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data.dataset import Dataset
from datasets import load_dataset
#-------------#
import librosa
import torchaudio
from transformers import AutoTokenizer
from typing import List
#-------------#


SAMPLE_RATE = 16000
HALF_BATCHSIZE_TIME = 2000


####################
# Sequence Dataset #
####################
class SequenceDataset(Dataset):
    
    def __init__(self, split, bucket_size, bucket_file, **kwargs):
        super(SequenceDataset, self).__init__()

        # self.ds = load_dataset("LIUM/tedlium", "release3", trust_remote_code=True)[split]
        self.ds = load_dataset("LIUM/tedlium", "release3")[split]
        self.sample_rate = SAMPLE_RATE
        self.tokenizer = AutoTokenizer.from_pretrained("facebook/wav2vec2-base-960h")

        # Read table for bucketing
        assert os.path.isdir(bucket_file), 'Please first run `python3 preprocess/preprocess_ted.py` to get bucket file.'

        # Wavs
        table_list = []
        file_path = os.path.join(bucket_file, split + ".csv")
        table_list = pd.read_csv(file_path)
        table_list = table_list.sort_values(by=['length'], ascending=False)

        X = table_list['file_path'].tolist()
        X_lens = table_list['length'].tolist()

        assert len(X) != 0, f"0 data found for {split}"

        # Transcripts
        self._load_transcript()

        # Use bucketing to allow different batch sizes at run time
        self.X = []
        batch_x, batch_len = [], []

        for x, x_len in tqdm(zip(X, X_lens), total=len(X), desc=f'ASR dataset {split}', dynamic_ncols=True):
            batch_x.append(x)
            batch_len.append(x_len)
                
            # Fill in batch_x until batch is full
            if len(batch_x) == bucket_size:
                # Half the batch size if seq too long
                if (bucket_size >= 2) and (max(batch_len) > HALF_BATCHSIZE_TIME):
                    self.X.append(batch_x[:bucket_size//2])
                    self.X.append(batch_x[bucket_size//2:])
                else:
                    self.X.append(batch_x)
                batch_x, batch_len = [], []
        
        # Gather the last batch
        if len(batch_x) > 0:
            self.X.append(batch_x)

    def _load_wav(self, idx) -> np.ndarray:
        instance = self.ds[idx]
        wav = librosa.resample(
            instance["audio"]["array"],
            orig_sr=self.ds.features["audio"].sampling_rate,
            target_sr=self.sample_rate
        )
        return wav
    
    def _load_text(self, idx) -> str:
        return self.ds[idx]["normalized_text"]
    
    def _load_label(self, idx) -> List[int]:
        return self.ds[idx]["text_ids"]

    def _load_transcript(self):
        def tokenize(batch):
            normalized_texts = []
            for text in batch["text"]:
                text = text.upper()
                text = text.replace(" '", "'")
                text = text.replace("-", " ")
                text = re.sub("[^ A-Z']", "", text)
                text = ' '.join(text.split())
                normalized_texts.append(text)
            outputs = self.tokenizer(
                normalized_texts,
                add_special_tokens=False,
                truncation=True,
                max_length=4096,
                padding=False,
            )
            return {"normalized_text": normalized_texts, "text_ids": outputs["input_ids"]}
        self.ds = self.ds.map(tokenize, batched=True, num_proc=4)  # use this in the finalized version
        # self.ds = self.ds.map(tokenize, batched=True, num_proc=4, load_from_cache_file=False)
        # print(self.ds[0])

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        # Load acoustic feature and pad
        wav_batch = [torch.from_numpy(self._load_wav(x_file)).float() for x_file in self.X[index]]
        label_batch = [self._load_label(x_file) for x_file in self.X[index]]
        text_batch = [self._load_text(x_file) for x_file in self.X[index]]
        filename_batch = [f"{x_file:07d}" for x_file in self.X[index]]
        return wav_batch, label_batch, text_batch, filename_batch

    def collate_fn(self, items):
        assert len(items) == 1
        return items[0][0], items[0][1], items[0][2], items[0][3]  # hack bucketing, return (wavs, labels, texts, filenames)
