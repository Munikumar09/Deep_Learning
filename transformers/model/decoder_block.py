import torch.nn as nn
from model.encoder_block import EncoderBlock
from model.self_attention import MultiHeadAttention


class DecoderBlock(nn.Module):
    def __init__(self, embed_size,batch_size,num_heads,mask,forward_expansion) -> None:
        super().__init__()
        self.embed_size=embed_size,
        self.batch_size=batch_size,
        self.masked_multi_head_attention=MultiHeadAttention(batch_size=batch_size,embed_size=embed_size,num_heads=num_heads,mask=mask)
        self.encoder_block=EncoderBlock(embed_size=embed_size,num_heads=num_heads,mask=mask,batch_size=batch_size,forward_expansion=forward_expansion)
        self.norm=nn.LayerNorm(embed_size)
    def forward(self,embeddings,encoder_outs):
        masked_attention=self.masked_multi_head_attention(embeddings,embeddings,embeddings)
        norm_attention=self.norm(embeddings+masked_attention)
        decoder_out=self.encoder_block(encoder_outs,encoder_outs,norm_attention)
        return decoder_out