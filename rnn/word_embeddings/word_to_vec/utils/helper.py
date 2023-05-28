import torch
import os
import yaml
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
import random
import numpy as np
from collections import Counter
from torch.nn import Embedding
from typing import Tuple,List
from torch import Tensor
from omegaconf import OmegaConf


def cosine_similarity(embedding:Embedding,device:str,n_valid_words:int=10,valid_window:int=100)->Tuple[Tensor,Tensor]:
    """
        Find the cosine similarity between the all the embedding and the valid words
        cosine similarity= a.b/|a||b|

    Parameters
    ----------
        embedding: ``Embedding``
            Embedding layer of the target words (input words)
        device: ``str``
            on which device to compute
        n_valid_words: ``int`` ( default = 10 )
            number of words to find cosine similarity
        valid_window: ``int`` ( default = 100 )
            window size to select the valid words

    Returns
    -------
        valid_words,similarities: ``Tuple[Tensor,Tensor]`` 
            randomly select valid words within the window size and cosine similarities values of all the embeddings with respect to the valid words
            
    """
    all_embeddings=embedding.weight
    magnitudes=all_embeddings.pow(2).sum(1).sqrt().unsqueeze(0)
    
    valid_words=random.sample(range(valid_window),n_valid_words//2)+random.sample(range(1000,1000+valid_window),n_valid_words//2)
    
    valid_words=torch.LongTensor(np.array(valid_words)).to(device)
    valid_embeddings=embedding(valid_words)
    similarities=torch.mm(valid_embeddings,all_embeddings.t())/magnitudes
    return valid_words,similarities

def get_noise_dist(int_words:List[int])->Tensor:
    """
    Calculate the probability distribution for selecting the negative samples
    with  (f(w)**(3/4))/(SUM(fw)**(3/4))

    Parameters
    ----------
        int_words: ``List[int]``
            List of integer representation of input tokens

    Returns
    -------
        noise_dists: ``Tensor`` 
            Noise distribution for selecting the negative samples
    """
    freq=Counter(int_words)
    freq_ratio={word:count/len(int_words) for word,count in freq.items()}
    freq_ratio=np.array(sorted(freq_ratio.values(),reverse=True))
    unigram_dist=freq_ratio/freq_ratio.sum()
    noise_dist=torch.from_numpy(unigram_dist**0.75/np.sum(unigram_dist**0.75))
    return noise_dist
    
    
def get_optim_class(optimizer_name:str)->optim.SGD or optim.Adam:
    """
    Get the optimizer class from the optimizer name

    Parameters
    ----------
        optimizer_name: ``str``
            name of the optimizer

    Returns
    -------
        optimizer_class: ``optim.SGD|optim.Adam`` 
            optimizer class
    """
    
    if optimizer_name.lower()=="adam":
        return  optim.Adam
    elif optimizer_name.lower()=="sgd":
        return  optim.SGD
    else:
        raise ValueError("Select optimizer from : sgd, adam")

def get_lr_scheduler(optimizer:optim.Adam or optim.SGD,total_epochs:int,verbose:bool=True):
    """
    Adjust the learning rate of optimizer for the total number of epochs.

    Parameters
    -------
        optimizer: ``optim.Adam|optim.SGD`` 
            optimizer set by the user
        total_epochs: ``int`` 
            total number of epochs
        verbose: ``bool`` ( default = True )
            whether to print the learning rate for each epoch

    Returns
    -------
        lr_schedular: ``LambdaLR`` 
            learning rate scheduler
    """
    lambd=lambda epoch: (total_epochs-epoch)/total_epochs
    lr_schedular=LambdaLR(optimizer=optimizer,lr_lambda=lambd,verbose=verbose)
    return lr_schedular

def save_config(config:dict,model_dir:str):
    """
    Save the config file which contains all the required parameters to run train and predict scripts.

    Parameters
    -------
        config: ``dict``
            dictionary of parameters required to train the model
        model_dir: ``str``
            path to save the configuration file
            
    """
    file_path=os.path.join(model_dir,"config.yaml")
    with open(file_path,"w") as stream:
        yaml.dump(config,stream)