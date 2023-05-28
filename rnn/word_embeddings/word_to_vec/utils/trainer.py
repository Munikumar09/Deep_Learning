import os
import torch
import json
import numpy as np
from tqdm import tqdm
from utils.helper import cosine_similarity
import random
from typing import Iterator, List,Tuple
from torch import optim

def get_context(int_words:List[int],idx:int,max_window_size:int)->List[int]:
    """
    Return context words within the max_window size from a list of integers (list of vocabulary indices), at an index  (target word index).
    
    Parameters:
    -----------
        int_words: ``List[int]`` 
            List of vocabulary indices.
        idx: `int`
            Target word index.
        max_window_size: ``int``
            Maximum window size.
        
    Returns:
    --------
        context_words:``List[int]``
            List of context words.
    """
    window_size=random.randint(1,max_window_size)
    start=max(0,idx-window_size)
    end=min(idx+window_size+1,len(int_words)-1)
    context_words=int_words[start:idx]+int_words[idx+1:end]
    return context_words

def get_data_iterator(words:List[int],batch_size:int,max_window_size:int)->Iterator[Tuple[torch.Tensor,torch.Tensor]]:
    """
    Create a dataset iterator that returns the right format for the model to be trained.
    
    Parameters:
    -----------
        words: ``List[int]``
            List of vocabulary indices.
        batch_size: ``int``
            Batch size.
        max_window_size: ``int`` 
            Maximum window size.
            
    Returns:
    --------
        data_iterator: ``Iterator[Tuple[torch.Tensor,torch.Tensor]]``
            Dataset iterator.

    """
    
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

class Trainer():
    def __init__(self,
                 model:torch.nn.Module,
                 train_data:List[int],
                 batch_size:int,
                 max_window_size:int,
                 int_to_vocab:dict[int,str],
                 epochs:int,
                 train_steps:int,
                 n_neg_samples:int,
                 checkpoint_frequency:int,
                 criterion:torch.nn.Module,
                 optimizer:optim,
                 lr_schedular:torch.optim.lr_scheduler,
                 device:str,
                 model_dir:str,
                 print_step:bool
         ) :
        self.model=model
        self.train_data=train_data
        self.batch_size=batch_size
        self.max_window_size=max_window_size
        self.int_to_vocab=int_to_vocab
        self.epochs=epochs
        self.train_steps=train_steps
        self.n_neg_samples=n_neg_samples
        self.checkpoint_frequency=checkpoint_frequency
        self.criterion=criterion
        self.optimizer=optimizer
        self.lr_schedular=lr_schedular
        self.device=device
        self.model_dir=model_dir
        self.print_step=print_step
        self.train_loss=[]
        self.model.to(self.device)
        
    
    def train(self):
        for epoch in range(self.epochs):
            self._train_epoch()
            if self.print_step is not None:
                print("Epoch: {}/{}".format((epoch+1), self.epochs))
                print("Loss: {:.4f}".format(self.train_loss[-1]))
                valid_idxs, similarities = cosine_similarity(self.model.target_embed,n_valid_words=10,valid_window=100,device=self.device)

                _, closest_idxs = similarities.topk(6)
                valid_idxs, closest_idxs = valid_idxs.to('cpu'), closest_idxs.to('cpu')
                
                for ii, v_idx in enumerate(valid_idxs):
                    closest_words = [self.int_to_vocab[idx.item()] for idx in closest_idxs[ii]][1:]
                    print(self.int_to_vocab[v_idx.item()] + " | "+ ", ".join(closest_words))
                print("\n...\n")
            self.lr_schedular.step()
            
            if self.checkpoint_frequency:
                self._save_checkpoint(epoch)
            
    def _train_epoch(self):
        self.model.train()
        running_loss=[]
        for i,batch in tqdm(enumerate(get_data_iterator(self.train_data,self.batch_size,self.max_window_size))):
            
            target_words=torch.LongTensor(batch[0]).to(self.device)
            context_words=torch.LongTensor(batch[1]).to(self.device)
            
            self.optimizer.zero_grad()
            target_embeddings=self.model.forward_target(target_words)
            context_embeddings=self.model.forward_context(context_words)
            noise_embeddings=self.model.forward_noise(target_words.shape[0],self.n_neg_samples)
            loss=self.criterion(target_embeddings,context_embeddings,noise_embeddings)
            loss.backward()
            self.optimizer.step()
            
            running_loss.append(loss.item())
            
            if i==self.train_steps:
                break
        epoch_loss=np.mean(running_loss)
        self.train_loss.append(epoch_loss)
    
    def _save_checkpoint(self,epoch:int):
        """
        save the model for given epochs

        Parameters
        ----------
            epoch: ``int``
                epoch value to save the model
        """
        epoch_num=epoch+1
        if epoch_num%self.epochs==0:
            model_path=f"checkpoint_{str(epoch).zfill(3)}"
            model_path=os.path.join(self.model_dir,model_path)
            torch.save(self.model,model_path)
    
    
    def save_model(self):
        model_path=os.path.join(self.model_dir,"model.pt")
        torch.save(self.model,model_path)
        
        
    def save_loss(self):
        loss_path=os.path.join(self.model_dir,"loss.json")
        with open(loss_path,"w") as fp:
            json.dump(self.train_loss,fp)
                