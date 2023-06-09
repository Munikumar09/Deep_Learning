import torch.nn as nn
from model.decoder_block import DecoderBlock

class Decoder(nn.Module):
    def __init__(
        self, embed_size, batch_size, num_heads, mask, forward_expansion, num_decoders,output_size,teacher_force_ratio
    ) -> None:
        super().__init__()
        self.embed_size = embed_size
        self.batch_size = batch_size
        self.num_heads = num_heads
        self.mask = mask
        self.forward_expansion = forward_expansion
        self.num_decoders = num_decoders
        self.teacher_force_ratio=teacher_force_ratio
        self.fc=nn.Linear(embed_size,output_size)

    def forward(self, embeddings, encoder_outs):
        decoder_block_outputs = embeddings
        
        for i in range(self.num_decoders):
            decoder_block = DecoderBlock(
                embed_size=self.embed_size,
                batch_size=self.batch_size,
                num_heads=self.num_heads,
                mask=self.mask,
                forward_expansion=self.forward_expansion,
            )
            decoder_block_outputs=decoder_block(embeddings,encoder_outs)
            
        outs=self.fc(decoder_block_outputs)
        return outs
        