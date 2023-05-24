import torch
import os
import yaml
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from utils.models import CBOWModel, SkipGramModel

def get_model_class(model_name):
    if model_name.lower()=="cbow":
        return CBOWModel
    elif model_name.lower()=="skipgram":
        return SkipGramModel
    else:
        raise ValueError("Select model from : cbow, skipgram")
    
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