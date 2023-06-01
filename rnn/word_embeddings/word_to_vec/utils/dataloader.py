import random
from collections import Counter
import numpy as np
from typing import List, Tuple


def preprocess_data(text: str) -> List[str]:
    """
    process the raw clean text data and return list of tokens

    Parameters
    ----------
        text: ``str``
            text data to process

    Returns
    -------
        filtered_words: ``List[str]``
            list of processed tokens
    """
    text = text.lower()
    words = text.split()
    word_counts = Counter(words)
    filtered_words = [word for word in words if word_counts[word] > 10]
    return filtered_words


def build_vocab(words: List[str]) -> Tuple[dict[str, int], dict[int, str]]:
    """
    Takes the input tokens and build the vocabulary from the tokens

    Parameters
    ----------
        words: ``List[str]``
            List of tokens to build the vocabulary

    Returns
    -------
        vocab_to_int,int_to_vocab: ``Tuple[dict[str,int],dict[int,str]]``
            vocab_to_int is a dictionary of vocab to int
            int_to_vocab is a dictionary of int_word to vocab
    """
    word_counts = Counter(words)
    sorted_vocab = sorted(word_counts, key=word_counts.get, reverse=True)
    vocab_to_int = {word: idx for idx, word in enumerate(sorted_vocab)}
    int_to_vocab = {idx: word for word, idx in vocab_to_int.items()}
    return vocab_to_int, int_to_vocab


def sub_sampling(int_words:List[int], threshold:float=1e-5)->List[int]:
    """ 
    It reduces the input tokes by eliminating the more frequent (stop words) like 'the','for' 
    because they won't provide any context by
        1-sqrt(threshold/freq_ratio(word)).

    Parameters
    ----------
        int_words: ``List[int]``
            list of integer representation of tokens
        threshold: ``float`` ( default = 1e-5 )
            threshold value to filter the tokens

    Returns
    -------
        sub_sampled_tokens: ``List[int]`` 
            list of int_words after sub sampling
    """
    word_counts = Counter(int_words)
    total_words = len(int_words)
    word_freq_ratios = {word: freq / total_words for word, freq in word_counts.items()}
    p_drop = {
        word: 1 - np.sqrt(threshold / word_freq_ratios[word]) for word in word_counts
    }
    sub_sampled_tokens= [word for word in int_words if random.random() > p_drop[word]]
    return sub_sampled_tokens

def load_data(data_path:str,subsample:bool):
    """
    It process data text data by converting to integer words and builds the vocabulary.

    Parameters
    ----------
        data_path: ``str``
            path to the data file
        subsample: ``bool``
            whether to subsample the data or not

    Returns
    -------
        int_words: ``List[int]``
            list of integer representation of tokens
        vocab_to_int,int_to_vocab: ``Tuple[dict[str,int],dict[int,str]]``
            vocab_to_int is a dictionary of vocab to int
            int_to_vocab is a dictionary of int_word to vocab
    """
    with open(data_path,'r') as fp:
        data=fp.read()
    data=preprocess_data(data)
    if subsample:
        data=sub_sampling(data)
    vocab_to_int,int_to_vocab=build_vocab(data)
    int_words=[vocab_to_int[word] for word in data]
    return int_words,vocab_to_int,int_to_vocab