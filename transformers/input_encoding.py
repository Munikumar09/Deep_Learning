import torch
import numpy as np
import torch.nn as nn


def get_positional_embeddings(input_shape,device):
    batch_size, seq_len, embed_size = input_shape
    pos=torch.arange(seq_len).reshape(seq_len,1).to(device)
    i=torch.arange(embed_size).to(device)
    positional_encodings=(pos/torch.pow(torch.Tensor([10000]).to(device),(2*i)/embed_size).expand((input_shape)))
    positional_encodings[:, :, 0::2] = torch.sin(positional_encodings[:, :, 0::2])
    positional_encodings[:, :, 1::2] = torch.cos(positional_encodings[:, :, 1::2])
    return positional_encodings


class InputEncoding(nn.Module):
    def __init__(self, embed_size, vocab_size,device) -> None:
        super().__init__()
    
        self.embed = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embed_size)
        self.device=device
    def forward(self, inputs):
        # inputs=seq_len,batch_size
        input_embeddings = self.embed(inputs)
        # input_embeddings=[seq_len,batch_size,embed_size]
        positional_embeddings = get_positional_embeddings(input_embeddings.shape,self.device)
        
        positional_encodings = input_embeddings + positional_embeddings
        return positional_encodings
