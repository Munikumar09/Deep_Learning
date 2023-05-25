import os
import torch
import json
import numpy as np
from tqdm import tqdm

class Trainer():
    def __init__(self,
                 model,
                 epochs,
                 train_dataloader,
                 train_steps,
                 val_dataloader,
                 val_steps,
                 checkpoint_frequency,
                 criterion,
                 optimizer,
                 lr_schedular,
                 device,
                 model_dir,
                #  model_name 
         ) :
        self.model=model
        self.epochs=epochs
        self.train_dataloader=train_dataloader
        self.train_steps=train_steps
        self.val_dataloader=val_dataloader
        self.val_steps=val_steps
        self.checkpoint_frequency=checkpoint_frequency
        self.criterion=criterion
        self.optimizer=optimizer
        self.lr_schedular=lr_schedular
        self.device=device
        self.model_dir=model_dir
        # self.model_name=model_name
        
        self.loss={"train":[],"val":[]}
        self.model.to(self.device)
        
    
    def train(self):
        for epoch in range(self.epochs):
            self._train_epoch()
            self._validation_epoch()
            print(
                f"{epoch+1} / {self.epochs} ====> \
            train loss : {self.loss['train'][-1]}\
            validation loss: {self.loss['val'][-1]}"
            )
            self.lr_schedular.step()
            
            if self.checkpoint_frequency:
                self._save_checkpoint(epoch)
    
    def _train_epoch(self):
        self.model.train()
        running_loss=[]
        for i,batch in tqdm(enumerate(self.train_dataloader)):
            inputs=batch[0].to(self.device)
            labels=batch[1].to(self.device)
            
            self.optimizer.zero_grad()
            
            outputs=self.model(inputs)
            loss=self.criterion(outputs,labels)
            loss.backward()
            self.optimizer.step()
            
            running_loss.append(loss.item())
            
            if i==self.train_steps:
                break
        epoch_loss=np.mean(running_loss)
        self.loss["train"].append(epoch_loss)
    def _validation_epoch(self):
        self.model.eval()
        running_loss=[]
        
        with torch.no_grad():
            for i,batch in enumerate(self.val_dataloader):
                inputs=batch[0].to(self.device)
                labels=batch[1].to(self.device)
                
                outputs=self.model(inputs)
                loss=self.criterion(outputs,labels)
                
                running_loss.append(loss.item())
                
                if i==self.val_steps:
                    break
        epoch_loss=np.mean(running_loss)
        self.loss["val"].append(epoch_loss)
    
    
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
            json.dump(self.loss,fp)
                