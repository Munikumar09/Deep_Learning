import torch.nn as nn
from model.encoder_block import EncoderBlock
from input_encoding import InputEncoding
from torch import Tensor


class Encoder(nn.Module):
    def __init__(
        self,
        embed_size: int,
        num_heads: int,
        forward_expansion: int,
        num_encoders: int,
        vocab_size: int,
        device: str,
    ) -> None:
        super().__init__()

        self.encodings = InputEncoding(embed_size, vocab_size, device)
        self.encoder_blocks = nn.ModuleList(
            [
                EncoderBlock(
                    embed_size=embed_size,
                    num_heads=num_heads,
                    forward_expansion=forward_expansion,
                )
                for _ in range(num_encoders)
            ]
        )

    def forward(self, embeddings: Tensor, mask: Tensor) -> Tensor:
        encoder_outputs = self.encodings(embeddings)
        for encoder_block in self.encoder_blocks:
            encoder_outputs = encoder_block(
                encoder_outputs, encoder_outputs, encoder_outputs, mask
            )
        return encoder_outputs
