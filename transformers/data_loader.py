import torch.nn as nn
import torch
from itertools import chain
from collections import Counter
from torchtext.vocab import vocab
from torch.utils.data import Dataset, random_split, DataLoader
import random
from torch.nn.utils.rnn import pad_sequence
from functools import partial
import os
import json
from tqdm import tqdm
from torchtext.data.metrics import bleu_score


def preprocess(data):
    data = data.replace("\u202f", " ").replace("\xa0", " ").replace("\u2009", " ")
    no_space = lambda char, prev_char: char in ",.!?" and prev_char != " "
    out = [
        " " + char if i > 0 and no_space(char, data[i - 1]) else char
        for i, char in enumerate(data)
    ]
    out = "".join(out)
    out = ["\t".join(sentence.split("\t")[:2]) for sentence in out.split("\n")]
    out = "\n".join(out)
    return out


def build_vocab(list_tokens):
    tokens = sorted(chain.from_iterable((list_tokens)))
    token_freq = Counter(tokens)
    vocabulary = vocab(token_freq, specials=["<unk>", "<pad>"])
    vocabulary.set_default_index(0)
    return vocabulary


def tokenizer(text):
    return [token for token in f"<sos> {text} <eos>".split(" ") if token]


def separate_src_tgt(data, max_samples=None):
    src = []
    tgt = []
    for i, text in enumerate(data):
        if max_samples and i > max_samples:
            break
        parts = text.split("\t")
        if len(parts) == 2:
            src.append(tokenizer(parts[0]))
            tgt.append(tokenizer(parts[1]))
    return src, tgt


class CustomDataset(Dataset):
    def __init__(self, dataset) -> None:
        super().__init__()
        self.src_data = dataset[0]
        self.tgt_data = dataset[1]

    def __len__(self):
        return len(self.src_data)

    def __getitem__(self, index):
        return self.src_data[index], self.tgt_data[index]


def train_test_split(dataset, train_percent):
    train_size = int(len(dataset) * train_percent)
    train_data = dataset[:train_size]
    test_data = dataset[train_size:]
    return train_data, test_data


def data_process_pipeline(data, eng_vocab=None, fra_vocab=None):
    src, tgt = separate_src_tgt(data)
    if eng_vocab is None:
        eng_vocab = build_vocab(src)
        fra_vocab = build_vocab(tgt)
    src_idx = [eng_vocab.forward(sent) for sent in src]
    tgt_idx = [fra_vocab.forward(sent) for sent in tgt]
    train_dataset = CustomDataset((src_idx, tgt_idx))
    return train_dataset, eng_vocab, fra_vocab


def load_data(data_path, train_percent):
    with open(data_path, "r", encoding="utf-8") as fp:
        data = fp.read()
    clean_data = preprocess(data)
    sent_list = [sent for sent in clean_data.split("\n") if len(sent) > 0]
    sorted_sent_list = sorted(sent_list, key=lambda x: len(x.split("\t")[0].split(" ")))
    train_data, test_data = train_test_split(sorted_sent_list, train_percent)
    return train_data, test_data


def collate_fn(train_data, src_pad_val, tgt_pad_val):
    src_data = [torch.LongTensor(src[0]) for src in train_data]
    tgt_data = [torch.LongTensor(tgt[1]) for tgt in train_data]
    src_tensor = pad_sequence(src_data, padding_value=src_pad_val)
    tgt_tensor = pad_sequence(tgt_data, padding_value=tgt_pad_val)
    return src_tensor, tgt_tensor
