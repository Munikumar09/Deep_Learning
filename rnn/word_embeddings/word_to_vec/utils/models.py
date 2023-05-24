import torch.nn as nn
from utils.constants import EMBED_DIMENSION, EMBED_MAX_NORM
from torchsummary import summary

class CBOWModel(nn.Module):
    def __init__(self, vocab_size: int):
        super(CBOWModel, self).__init__()
        self.embedding = nn.Embedding(
            vocab_size, 
            EMBED_DIMENSION, 
            max_norm=EMBED_MAX_NORM
        )
        self.linear=nn.Linear(in_features=EMBED_DIMENSION,out_features=vocab_size)
    def forward(self,inputs):
        outputs=self.embedding(inputs)
        outputs=outputs.mean(axis=1)
        outputs=self.linear(outputs)
        return outputs
    

class SkipGramModel(nn.Module):
    def __init__(self, vocab_size:int):
        super(SkipGramModel,self).__init__()
        self.embedding=nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=EMBED_DIMENSION,
            max_norm=EMBED_MAX_NORM
        )
        self.linear=nn.Linear(in_features=EMBED_DIMENSION,
                          out_features=vocab_size)
    def forward(self,inputs):
        embeddings=self.embedding(inputs)
        output=self.linear(embeddings)
        return output