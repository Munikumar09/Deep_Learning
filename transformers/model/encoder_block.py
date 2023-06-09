import torch.nn as nn
from model.self_attention import MultiHeadAttention

class EncoderBlock(nn.Module):
    def __init__(self, embed_size,num_heads,forward_expansion) -> None:
        super().__init__()
        self.multi_head_attention=MultiHeadAttention(embed_size=embed_size,num_heads=num_heads)
        self.layer_norm1=nn.LayerNorm(embed_size)
        self.fc1=nn.Linear(embed_size,embed_size*forward_expansion)
        self.relu=nn.ReLU()
        self.fc2=nn.Linear(embed_size*forward_expansion,embed_size)
        self.layer_norm2=nn.LayerNorm(embed_size)
    
    def forward(self,queries,keys,values,mask,dec=False):
        attention=self.multi_head_attention(queries,keys,values,mask,dec)
        attention=self.layer_norm1(queries+attention)
        fc=self.fc1(attention)
        fc=self.relu(fc)
        fc=self.fc2(fc)
        output=self.layer_norm2(attention+fc)
        return output