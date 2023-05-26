import os
import torch
import torch.nn as nn
from utils.helper import get_optim_class,get_lr_scheduler,save_config,save_vocab
from utils.trainer import Trainer
from utils.dataloader import get_data_iterator,load_data
import hydra
from hydra.core.config_store import ConfigStore
from config import WordToVec
from utils.models import SkipGramNegSampling,NegativeSamplingLoss
from utils.helper import get_noise_dist
cs= ConfigStore.instance()
cs.store(name="word_to_vec", node=WordToVec)

@hydra.main(config_path=".",config_name="config.yaml",version_base=None)
def train(cfg:WordToVec):
    device="cuda" if torch.cuda.is_available() else "cpu"
    if not os.path.exists(cfg.paths.model_dir):
        os.makedirs(cfg.paths.model_dir)
    train_data,vocab_to_int,int_to_vocab=load_data(cfg.paths.data_path,True)
    dataset_iterator=get_data_iterator(train_data,cfg.params.batch_size,4)
    
    noise_dist=get_noise_dist(train_data)
    
    model=SkipGramNegSampling(len(int_to_vocab),cfg.params.embed_size,noise_dist,device)
    
    criterion=NegativeSamplingLoss()
    
    optimizer_class=get_optim_class(cfg.params.optimizer)
    optimizer=optimizer_class(model.parameters(),cfg.params.learning_rate)
    
    lr_schedular=get_lr_scheduler(optimizer=optimizer,total_epochs=cfg.params.epochs)
    
    trainer=Trainer(
        model=model,
        dataset_iter=dataset_iterator,
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
    save_vocab(vocab=vocab_to_int,model_dir=cfg.paths.model_dir)
    print(f"model artifacts are saved to {cfg.paths.model_dir}")
    
if __name__=="__main__":
    train()