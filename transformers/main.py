from data.data_loader import load_data, data_process_pipeline
from data.helper import collate_fn,predict_pipeline
from functools import partial
from torch.utils.data import DataLoader
import torch
from trainer import Trainer
from model.transformer_encoder import Encoder
from model.transformer_decoder import Decoder
from model.transformer import Transformer
import torch.nn as nn
device = "cuda" if torch.cuda.is_available() else "cpu"

# hyperparamers
num_epochs=2
learning_rate=0.001
batch_size = 512
encoder_embedding_size = 512
decoder_embedding_size = 512
hidden_size = 1024
encoder_n_layers = 2
decoder_n_layers = 2
encoder_dropout = 0.5
decoder_dropout = 0.5
teacher_forcing_ratio = 0.5
num_heads = 8
forward_expansion = 4
num_encoders = 6


raw_train_data, raw_val_data = load_data(data_path="fra.txt", train_percent=0.8)
train_data, eng_vocab, fra_vocab = data_process_pipeline(raw_train_data)
val_data, _, _ = data_process_pipeline(
    raw_val_data, eng_vocab=eng_vocab, fra_vocab=fra_vocab
)

# data loader
train_loader = DataLoader(
    train_data,
    batch_size=batch_size,
    collate_fn=partial(
        collate_fn, src_pad_val=eng_vocab["<pad>"], tgt_pad_val=fra_vocab["<pad>"]
    ),
)
val_loader = DataLoader(
    val_data,
    batch_size=batch_size,
    collate_fn=partial(
        collate_fn, src_pad_val=eng_vocab["<pad>"], tgt_pad_val=fra_vocab["<pad>"]
    ),
)

# vocab sizes
input_size_encoder = len(eng_vocab)
input_size_decoder = len(fra_vocab)
output_size_decoder = len(fra_vocab)

encoder = Encoder(
    embed_size=encoder_embedding_size,
    num_heads=num_heads,
    forward_expansion=forward_expansion,
    num_encoders=num_encoders,
    vocab_size=len(eng_vocab),
)

decoder = Decoder(
    embed_size=decoder_embedding_size,
    num_heads=num_heads,
    forward_expansion=forward_expansion,
    num_decoders=num_encoders,
    output_size=len(fra_vocab),
    vocab_size=len(fra_vocab),
)
transformer = Transformer(
    encoder=encoder,
    decoder=decoder,
    src_pad_idx=eng_vocab["<pad>"],
    tgt_pad_idx=fra_vocab["<pad>"],
    device=device,
)

criterion = nn.CrossEntropyLoss(ignore_index=fra_vocab['<pad>'])
optimizer=torch.optim.Adam(transformer.parameters(),lr=learning_rate)
trainer=Trainer(model=transformer,
            num_epochs=num_epochs,
            batch_size=batch_size,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            print_stats=True,
            tgt_vocab=fra_vocab,
            src_vocab=eng_vocab,
            text_pipeline=predict_pipeline
        )
trainer.train(train_loader=train_loader,val_loader=val_loader,save_path="results")