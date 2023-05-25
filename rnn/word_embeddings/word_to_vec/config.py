from dataclasses import dataclass

@dataclass
class Params:
    model_name:str
    dataset:str
    train_batch_size:int
    val_batch_size:int
    shuffle:bool
    optimizer:str
    learning_rate:float
    epochs:int
    train_steps:int
    val_steps:int
    checkpoint_frequency:int

@dataclass
class Paths:
    model_dir:str
    data_dir:str
@dataclass
class WordToVec:
    params:Params
    paths:Paths