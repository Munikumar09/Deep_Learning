import torch
import os
import yaml
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
import random
import numpy as np
from collections import Counter
def cosine_similarity(embeddings,n_valid_words=10,valid_window=100,device="cpu"):
    all_embeddings=embeddings.weight
    magnitudes=all_embeddings.pow(2).sum(1).sqrt().unsqueeze(0)
    valid_words=random.sample(range(valid_window),n_valid_words//2)+random.sample(range(1000,1000+valid_window),n_valid_words//2)
    valid_words=torch.LongTensor(np.array(valid_words)).to(device)
    
    similarities=torch.mm(valid_words,all_embeddings.t())/magnitudes
    return valid_words,similarities

def get_noise_dist(int_words):
    freq=Counter(int_words)
    freq_ratio={word:count/len(int_words) for word,count in freq.items()}
    freq_ratio=np.array(sorted(freq_ratio.values(),reverse=True))
    unigram_dist=freq_ratio/freq_ratio.sum()
    noise_dist=torch.from_numpy(unigram_dist**0.75/np.sum(unigram_dist**0.75))
    return noise_dist
    
    
def get_optim_class(optimizer_name):
    if optimizer_name.lower()=="adam":
        return  optim.Adam
    elif optimizer_name.lower()=="sgd":
        return  optim.SGD
    else:
        raise ValueError("Select optimizer from : sgd, adam")

def get_lr_scheduler(optimizer,total_epochs,verbose=True):
    lambd=lambda epoch: (total_epochs-epoch)/total_epochs
    lr_schedular=LambdaLR(optimizer=optimizer,lr_lambda=lambd,verbose=verbose)
    return lr_schedular

def save_config(config,model_dir):
    file_path=os.path.join(model_dir,"config.yaml")
    with open(file_path,"w") as stream:
        yaml.dump(config,stream)

def save_vocab(vocab,model_dir):
    file_path=os.path.join(model_dir,"vocab.pt")
    torch.save(vocab,file_path)