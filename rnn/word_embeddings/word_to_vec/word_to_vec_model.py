import os
import torch
import torch.nn as nn
from utils.helper import get_optim_class,get_lr_scheduler,save_config
from utils.trainer import Trainer
import hydra
from hydra.core.config_store import ConfigStore
from config import WordToVec
from utils.models import SkipGramNegSampling,NegativeSamplingLoss
from utils.helper import get_noise_dist
from utils.dataloader import load_data
from omegaconf import OmegaConf
import yaml
import json
cs= ConfigStore.instance()
cs.store(name="word_to_vec", node=WordToVec)

@hydra.main(config_path=".",config_name="config.yaml",version_base=None)
def train(cfg:WordToVec):
    with open("conff.yaml",'w')as stream:
        OmegaConf.save(cfg,stream)
    return
    device="cuda" if torch.cuda.is_available() else "cpu"
    if not os.path.exists(cfg.paths.model_dir):
        os.makedirs(cfg.paths.model_dir)
    train_data,vocab_to_int,int_to_vocab=load_data(cfg.paths.data_path,True)
    noise_dist=get_noise_dist(train_data)
    
    model=SkipGramNegSampling(len(int_to_vocab),cfg.params.embed_size,noise_dist,device)
    
    criterion=NegativeSamplingLoss()
    
    optimizer_class=get_optim_class(cfg.params.optimizer)
    optimizer=optimizer_class(model.parameters(),cfg.params.learning_rate)
    
    lr_schedular=get_lr_scheduler(optimizer=optimizer,total_epochs=cfg.params.epochs)
    
    trainer=Trainer(
        model=model,
        train_data=train_data,
        batch_size=cfg.params.batch_size,
        max_window_size=cfg.params.max_window_size,
        int_to_vocab=int_to_vocab,
        epochs=cfg.params.epochs,
        train_steps=cfg.params.train_steps,
        n_neg_samples=cfg.params.n_neg_samples,
        checkpoint_frequency=cfg.params.checkpoint_frequency,
        criterion=criterion,
        optimizer=optimizer,
        lr_schedular=lr_schedular,
        device=device,
        model_dir=cfg.paths.model_dir,
        print_step=cfg.params.print_step
        
    )
    print('Training started...')
    trainer.train()
    print("Training finished.")
    trainer.save_model()
    trainer.save_loss()
    save_config(config=cfg,model_dir=cfg.paths.model_dir)
    print(f"model artifacts are saved to {cfg.paths.model_dir}")
    
if __name__=="__main__":
    train()