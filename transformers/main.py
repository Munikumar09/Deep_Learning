from data_loader import load_data,collate_fn,data_process_pipeline
from functools import partial
from torch.utils.data import DataLoader
import torch

device="cuda" if torch.cuda.is_available() else "cpu"

#hyperparamers
batch_size=512
encoder_embedding_size=300
decoder_embedding_size=300
hidden_size=1024
encoder_n_layers=2
decoder_n_layers=2
encoder_dropout=0.5
decoder_dropout=0.5
teacher_forcing_ratio=0.5


raw_train_data,raw_val_data=load_data(data_path="fra.txt",train_percent=0.8)
train_data,eng_vocab,fra_vocab=data_process_pipeline(raw_train_data)
val_data,_,_=data_process_pipeline(raw_val_data,eng_vocab=eng_vocab,fra_vocab=fra_vocab)

#data loader
train_loader = DataLoader(train_data, batch_size=batch_size, collate_fn=partial(collate_fn,src_pad_val=eng_vocab['<pad>'],tgt_pad_val=fra_vocab['<pad>']))
val_loader=DataLoader(val_data,batch_size=batch_size,collate_fn=partial(collate_fn,src_pad_val=eng_vocab['<pad>'],tgt_pad_val=fra_vocab['<pad>']))

#vocab sizes
input_size_encoder=len(eng_vocab)
input_size_decoder=len(fra_vocab)
output_size_decoder=len(fra_vocab)

print(len(train_loader))
print(len(val_loader))