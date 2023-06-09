import torch
import numpy as np
import torch.nn as nn


def get_positional_embeddings(input_shape):
    batch_size, seq_len, embed_size = input_shape
    positional_encoding = [
        pos / 10000 ** ((2 * i) / embed_size)
        for batch in range(batch_size)
        for pos in range(seq_len)
        for i in range(embed_size)
    ]
    positional_encoding = np.array(positional_encoding).reshape(input_shape)

    positional_encoding = torch.from_numpy(positional_encoding).float()
    positional_encoding[:, :, 0::2] = torch.sin(positional_encoding[:, :, 0::2])
    positional_encoding[:, :, 1::2] = torch.cos(positional_encoding[:, :, 1::2])
    return positional_encoding


class InputEncoding(nn.Module):
    def __init__(self, embed_size, vocab_size) -> None:
        super().__init__()
        self.embed = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embed_size)

    def forward(self, inputs):
        # inputs=seq_len,batch_size
        input_embeddings = self.embed(inputs)
        # input_embeddings=[seq_len,batch_size,embed_size]
        positional_embeddings = get_positional_embeddings(input_embeddings.shape)
        positional_encodings = input_embeddings + positional_embeddings
        return positional_encodings
