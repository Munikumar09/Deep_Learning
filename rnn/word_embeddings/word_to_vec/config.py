from dataclasses import dataclass

@dataclass
class Params:
    model_name:str
    dataset:str
    batch_size:int
    shuffle:bool
    optimizer:str
    learning_rate:float
    epochs:int
    train_steps:int
    embed_size:int
    checkpoint_frequency:int
    n_neg_samples:int
    print_step:int

@dataclass
class Paths:
    model_dir:str
    data_dir:str
    data_path:str
@dataclass
class WordToVec:
    params:Params
    paths:Paths