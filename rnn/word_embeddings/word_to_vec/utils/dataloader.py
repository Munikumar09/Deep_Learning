import random
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

def get_context(int_words,idx,max_window_size):
    window_size=random.randint(1,max_window_size)
    start=max(0,idx-window_size)
    end=min(idx+window_size+1,len(int_words)-1)
    context_words=int_words[start:idx]+int_words[idx+1:end]
    return context_words

def load_data(data_path,sampling):
    with open(data_path,"r") as fp:
        data=fp.read()
    data=preprocess_data(data)
    if sampling:
        data=sub_sampling(data)
    vocab_to_int,int_to_vocab=build_vocab(data)
    int_words=[vocab_to_int[word] for word in data]
    return int_words,vocab_to_int,int_to_vocab

def get_data_iterator(words,batch_size,max_window_size):
    n_batches=len(words)//batch_size
    words=words[:n_batches*batch_size]
    for batch_num in range(0,len(words),batch_size):
        batch_words=words[batch_num:batch_num+batch_size]
        target_batch,context_batch=[],[]
        
        for i in range(len(batch_words)):
            target_word=[batch_words[i]]
            context_words=get_context(batch_words,i,max_window_size)
            
            target_batch.extend(target_word*len(context_words))
            context_batch.extend(context_words)
        yield target_batch,context_batch