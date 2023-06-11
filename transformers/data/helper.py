import torch
from torch.nn.utils.rnn import pad_sequence
from itertools import chain
from collections import Counter
from torchtext.vocab import vocab, Vocab
from typing import List, Tuple, Union
from torch import Tensor


def collate_fn(data, src_pad_val, tgt_pad_val):
    src_data = [torch.LongTensor(src[0]) for src in data]
    tgt_data = [torch.LongTensor(tgt[1]) for tgt in data]
    src_tensor = pad_sequence(src_data, padding_value=src_pad_val)
    tgt_tensor = pad_sequence(tgt_data, padding_value=tgt_pad_val)
    return src_tensor, tgt_tensor


def build_vocab(list_tokens: List[List[str]]) -> Vocab:
    tokens = sorted(chain.from_iterable((list_tokens)))
    token_freq = Counter(tokens)
    vocabulary = vocab(token_freq, specials=["<unk>", "<pad>"])
    vocabulary.set_default_index(0)
    return vocabulary


def preprocess(data: str) -> str:
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


def tokenizer(text: str) -> List[str]:
    return [token for token in f"<sos> {text} <eos>".split(" ") if token]


def separate_src_tgt(
    data: List[str], max_samples: int = None
) -> Tuple[List[List[str]], List[List[str]]]:
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


def train_test_split(
    dataset: List[str], train_percent: float
) -> Tuple[List[str], List[str]]:
    train_size = int(len(dataset) * train_percent)
    train_data = dataset[:train_size]
    test_data = dataset[train_size:]
    return train_data, test_data


def predict_pipeline(txt: Union[str, List[str]], eng_vocab: Vocab)->Tensor:
    if isinstance(txt, str):
        txt = [txt]
    sent_tokens = [tokenizer(tokens) for tokens in txt]
    int_tokens = [eng_vocab.forward(tokens) for tokens in sent_tokens]
    src_tensor = [torch.LongTensor(token_list) for token_list in int_tokens]
    src = pad_sequence(src_tensor, padding_value=eng_vocab["<pad>"])
    return src
