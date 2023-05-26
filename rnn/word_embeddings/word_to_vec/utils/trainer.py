import os
import torch
import json
import numpy as np
from tqdm import tqdm
from utils.helper import cosine_similarity

class Trainer():
    def __init__(self,
                 model,
                 dataset_iter,
                 int_to_vocab,
                 epochs,
                 train_steps,
                 n_neg_samples,
                 checkpoint_frequency,
                 criterion,
                 optimizer,
                 lr_schedular,
                 device,
                 model_dir,
                 print_step
         ) :
        self.model=model
        self.dataset_iter=dataset_iter
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
                valid_idxs, similarities = cosine_similarity(self.model.target_embed)
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
        for i,batch in tqdm(enumerate(self.dataset_iter)):
            
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
    
    def _save_checkpoint(self,epoch):
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
                