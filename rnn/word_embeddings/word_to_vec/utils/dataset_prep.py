from torch.utils.data import Dataset
from collections import Counter
import numpy as np
import random

def preprocess_data(text:str):
    text=text.lower()
    words=text.split()
    word_counts=Counter(words)
    filtered_words=[word for word in words if word_counts[word]>10]
    return filtered_words

def build_vocab(words):
    word_counts=Counter(words)
    sorted_vocab=sorted(word_counts,key=word_counts.get,reverse=True)
    word_to_int={word:idx for idx,word in enumerate(sorted_vocab)}
    int_to_word={idx:word for word,idx in word_to_int.items()}
    return word_to_int,int_to_word

def sub_sampling(int_words,threshold=1e-5):
    word_counts=Counter(int_words)
    total_words=len(int_words)
    word_freq_ratios={word:freq/total_words for word,freq in word_counts.items()}   
    p_drop={word:1-np.sqrt(threshold/word_freq_ratios[word]) for word in word_counts}
    rand_prob=random.random()
    return [word for word in int_words if rand_prob>p_drop[word]]

class CustomTextData(Dataset):
    def __init__(self,data_path,data_preprocess,sub_sampling):
        super().__init__()
        with open(data_path,'r') as fp:
            data=fp.read()
        if data_preprocess:
            data=preprocess_data(data)
        self.vocab_to_int,self.int_to_vocab=build_vocab(data)
        self.int_words=[self.vocab_to_int[word] for word in data]
        if sub_sampling==True:
            int_words=sub_sampling(self.int_words)
    def __len__(self):
        return len(self.int_words)
    def __getitem__(self,idx):
        return self.int_words[idx]