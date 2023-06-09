import torch.nn as nn
from model.encoder_block import EncoderBlock


class Encoder(nn.Module):
    def __init__(
        self, embed_size, num_heads, mask, batch_size, forward_expansion, num_encoders
    ) -> None:
        super().__init__()
        self.num_encoders = num_encoders
        self.embed_size = embed_size
        self.num_heads = num_heads
        self.mask = mask
        self.batch_size = batch_size
        self.forward_expansion = forward_expansion

    def forward(self, embeddings):
        encoder_outputs = embeddings
        for i in range(self.num_encoders):
            encoder_block = EncoderBlock(
                embed_size=self.embed_size,
                num_heads=self.num_heads,
                mask=self.mask,
                batch_size=self.batch_size,
                forward_expansion=self.forward_expansion,
            )
            encoder_outputs = encoder_block(encoder_outputs,encoder_outputs,encoder_outputs)
        return encoder_outputs
