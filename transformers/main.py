from data.data_loader import load_data, data_process_pipeline
from data.helper import collate_fn, predict_pipeline
from functools import partial
from torch.utils.data import DataLoader
import torch
from trainer import Trainer
from model.transformer_encoder import Encoder
from model.transformer_decoder import Decoder
from model.transformer import Transformer
import torch.nn as nn
from utils.helper import display_info, get_bleu_score

device = "cuda" if torch.cuda.is_available() else "cpu"

# hyperparamers
num_epochs = 25
learning_rate = 0.01
batch_size = 256
encoder_embedding_size = 512
decoder_embedding_size = 512
encoder_dropout = 0.5
decoder_dropout = 0.5
num_heads = 8
forward_expansion = 4
num_encoders = 6
num_decoders = 6
num_workers = 4
train_percent = 0.8
data_path = "fra.txt"


def train():
    # loading the raw training data
    raw_train_data, raw_val_data = load_data(
        data_path=data_path, train_percent=train_percent
    )

    # preparing the data
    train_data, eng_vocab, fra_vocab = data_process_pipeline(raw_train_data)
    val_data, _, _ = data_process_pipeline(
        raw_val_data, eng_vocab=eng_vocab, fra_vocab=fra_vocab
    )

    # data loader
    train_loader = DataLoader(
        train_data,
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=partial(
            collate_fn, src_pad_val=eng_vocab["<pad>"], tgt_pad_val=fra_vocab["<pad>"]
        ),
    )
    val_loader = DataLoader(
        val_data,
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=partial(
            collate_fn, src_pad_val=eng_vocab["<pad>"], tgt_pad_val=fra_vocab["<pad>"]
        ),
    )

    encoder = Encoder(
        embed_size=encoder_embedding_size,
        num_heads=num_heads,
        forward_expansion=forward_expansion,
        num_encoders=num_encoders,
        vocab_size=len(eng_vocab),
        device=device,
    )

    decoder = Decoder(
        embed_size=decoder_embedding_size,
        num_heads=num_heads,
        forward_expansion=forward_expansion,
        num_decoders=num_decoders,
        output_size=len(fra_vocab),
        vocab_size=len(fra_vocab),
        device=device,
    )
    model = Transformer(
        encoder=encoder,
        decoder=decoder,
        src_pad_idx=eng_vocab["<pad>"],
        tgt_pad_idx=fra_vocab["<pad>"],
        device=device,
    )
    partial_dispaly_info = partial(
        display_info,
        text_pipeline=predict_pipeline,
        src_vocab=eng_vocab,
        tgt_vocab=fra_vocab,
    )
    partial_bleu_score = partial(get_bleu_score, fra_vocab)

    criterion = nn.CrossEntropyLoss(ignore_index=fra_vocab["<pad>"])
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    trainer = Trainer(
        model=model,
        num_epochs=num_epochs,
        batch_size=batch_size,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        print_stats=True,
        display_info=partial_dispaly_info,
        bleu_score=partial_bleu_score,
    )
    trainer.train(train_loader=train_loader, val_loader=val_loader, save_path="results")
    return model


if __name__ == "__main__":
    model = train()
