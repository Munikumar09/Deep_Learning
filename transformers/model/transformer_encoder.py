import torch.nn as nn
from model.encoder_block import EncoderBlock
from input_encoding import InputEncoding

class Encoder(nn.Module):
    def __init__(
        self, embed_size, num_heads,forward_expansion, num_encoders,vocab_size
    ) -> None:
        super().__init__()
        self.num_encoders = num_encoders
        self.embed_size = embed_size
        self.num_heads = num_heads
        
        self.forward_expansion = forward_expansion
        self.encodings=InputEncoding(embed_size,vocab_size)

    def forward(self, embeddings,mask):
        encoder_outputs = self.encodings(embeddings)
        for i in range(self.num_encoders):
            encoder_block = EncoderBlock(
                embed_size=self.embed_size,
                num_heads=self.num_heads,
                forward_expansion=self.forward_expansion
                
            )
            encoder_outputs = encoder_block(encoder_outputs,encoder_outputs,encoder_outputs,mask)
        return encoder_outputs
