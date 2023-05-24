import os
import torch
import yaml
import argparse
import torch.nn as nn
from utils.helper import get_model_class,get_optim_class,get_lr_scheduler,save_config,save_vocab
from utils.trainer import Trainer
from utils.dataloader import get_dataloader_and_vocab

def train(config):
    if not os.path.exists(config["model_dir"]):
        os.makedirs(config["model_dir"])
    
    train_dataloader,vocab=get_dataloader_and_vocab(
        model_name=config["model_name"],
        dataset_name=config["dataset"],
        split_type="train",
        data_dir=config['data_dir'],
        batch_size=config["train_batch_size"],
        shuffle=config["shuffle"],
        vocab=None
    )
    
    val_dataloader,_=get_dataloader_and_vocab(
        model_name=config["model_name"],
        dataset_name=config["dataset"],
        split_type="valid",
        data_dir=config["data_dir"],
        batch_size=config["val_batch_size"],
        shuffle=False,
        vocab=vocab
    )
    
    vocab_size=len(vocab)
    
    model_class=get_model_class(config["model_name"])
    
    model=model_class(vocab_size=vocab_size)
    
    criterion=nn.CrossEntropyLoss()
    
    optimizer_class=get_optim_class(config["optimizer"])
    optimizer=optimizer_class(model.parameters(),config["learning_rate"])
    
    lr_schedular=get_lr_scheduler(optimizer=optimizer,total_epochs=config["epochs"])
    
    device="cuda" if torch.cuda.is_available() else "cpu"
    
    trainer=Trainer(
        model=model,
        epochs=config["epochs"],
        train_dataloader=train_dataloader,
        train_steps=config["train_steps"],
        val_dataloader=val_dataloader,
        val_steps=config["val_steps"],
        checkpoint_frequency=config["checkpoint_frequency"],
        criterion=criterion,
        device=device,
        optimizer=optimizer,
        lr_schedular=lr_schedular,
        model_dir=config["model_dir"]
    )
    print('Training started...')
    trainer.train()
    print("Training finished.")
    trainer.save_model()
    trainer.save_loss()
    save_config(config=config,model_dir=config["model_dir"])
    save_vocab(vocab=vocab,model_dir=config["model_dir"])
    
    print(f"model artifacts are saved to {config['model_dir']}")
    
if __name__=="__main__":
    argparse=argparse.ArgumentParser()
    argparse.add_argument("--config",type=str, required=True,help="path the config yaml file")
    args=argparse.parse_args()
    
    with open(args.config,'r') as stream:
        config=yaml.safe_load(stream)
    train(config)