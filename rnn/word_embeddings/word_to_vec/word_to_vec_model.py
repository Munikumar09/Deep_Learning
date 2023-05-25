import os
import torch
import yaml
import argparse
import torch.nn as nn
from utils.helper import get_model_class,get_optim_class,get_lr_scheduler,save_config,save_vocab
from utils.trainer import Trainer
from utils.dataloader import get_dataloader_and_vocab
import hydra
from hydra.core.config_store import ConfigStore
from config import WordToVec

cs= ConfigStore.instance()
cs.store(name="word_to_vec", node=WordToVec)

@hydra.main(config_path=".",config_name="config.yaml",version_base=None)
def train(cfg:WordToVec):
    if not os.path.exists(cfg.paths.model_dir):
        os.makedirs(cfg.paths.model_dir)
    
    train_dataloader,vocab=get_dataloader_and_vocab(
        model_name=cfg.params.model_name,
        dataset_name=cfg.params.dataset,
        split_type="train",
        data_dir=cfg.paths.data_dir,
        batch_size=cfg.params.train_batch_size,
        shuffle=cfg.params.shuffle,
        vocab=None
    )
    
    val_dataloader,_=get_dataloader_and_vocab(
        model_name=cfg.params.model_name,
        dataset_name=cfg.params.dataset,
        split_type="valid",
        data_dir=cfg.paths.data_dir,
        batch_size=cfg.params.val_batch_size,
        shuffle=False,
        vocab=vocab
    )
    
    vocab_size=len(vocab)
    
    model_class=get_model_class(cfg.params.model_name)
    
    model=model_class(vocab_size=vocab_size)
    
    criterion=nn.CrossEntropyLoss()
    
    optimizer_class=get_optim_class(cfg.params.optimizer)
    optimizer=optimizer_class(model.parameters(),cfg.params.learning_rate)
    
    lr_schedular=get_lr_scheduler(optimizer=optimizer,total_epochs=cfg.params.epochs)
    
    device="cuda" if torch.cuda.is_available() else "cpu"
    
    trainer=Trainer(
        model=model,
        epochs=cfg.params.epochs,
        train_dataloader=train_dataloader,
        train_steps=cfg.params.train_steps,
        val_dataloader=val_dataloader,
        val_steps=cfg.params.val_steps,
        checkpoint_frequency=cfg.params.checkpoint_frequency,
        criterion=criterion,
        device=device,
        optimizer=optimizer,
        lr_schedular=lr_schedular,
        model_dir=cfg.paths.model_dir
    )
    print('Training started...')
    trainer.train()
    print("Training finished.")
    trainer.save_model()
    trainer.save_loss()
    save_config(config=config,model_dir=cfg.paths.model_dir)
    save_vocab(vocab=vocab,model_dir=cfg.paths.model_dir)
    
    print(f"model artifacts are saved to {cfg.paths.model_dir}")
    
if __name__=="__main__":
    train()