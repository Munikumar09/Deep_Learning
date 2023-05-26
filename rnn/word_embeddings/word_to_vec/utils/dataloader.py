from torch.utils.data import DataLoader
from functools import partial
import random
 
import torch

from torch.utils.data import Dataset
from collections import Counter
import numpy as np


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

def get_context(int_words,idx,window_size):
    start=max(0,idx-window_size)
    end=min(idx+window_size+1,len(int_words)-1)
    context_words=int_words[start:idx]+int_words[idx+1:end]
    return context_words

class CustomTextData(Dataset):
    def __init__(self,data_path,data_preprocess,sub_sampling,max_window_size):
        super().__init__()
        self.max_window_size=max_window_size
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
        target_word=self.int_words[idx]
        context_words=get_context(self.int_words,idx,self.max_window_size)
        target_tensor=torch.tensor([target_word]*len(context_words))
        context_tensor=torch.tensor(context_words)
        return target_tensor,context_tensor

def get_dataloader(dataset,batch_size,shuffle):
  
    dataloader=DataLoader(
        dataset,
        batch_size,
        shuffle,
    )
    return dataloader
