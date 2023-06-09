import torch.nn as nn
from model.encoder_block import EncoderBlock
from model.self_attention import MultiHeadAttention


class DecoderBlock(nn.Module):
    def __init__(self, embed_size,num_heads,forward_expansion) -> None:
        super().__init__()
        self.embed_size=embed_size,
        
        self.masked_multi_head_attention=MultiHeadAttention(embed_size=embed_size,num_heads=num_heads)
        self.encoder_block=EncoderBlock(embed_size=embed_size,num_heads=num_heads,forward_expansion=forward_expansion)
        self.norm=nn.LayerNorm(embed_size)
    def forward(self,embeddings,encoder_outs,mask):
        masked_attention=self.masked_multi_head_attention(embeddings,embeddings,embeddings,mask)
        norm_attention=self.norm(embeddings+masked_attention)
        decoder_out=self.encoder_block(norm_attention,encoder_outs,encoder_outs,None)
        return decoder_out