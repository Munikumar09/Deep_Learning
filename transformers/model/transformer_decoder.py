import torch.nn as nn
from model.decoder_block import DecoderBlock
from input_encoding import InputEncoding
from torch import Tensor


class Decoder(nn.Module):
    def __init__(
        self,
        embed_size: int,
        num_heads: int,
        forward_expansion: int,
        num_decoders: int,
        output_size: int,
        vocab_size: int,
        device: str,
    ) -> None:
        super().__init__()
        
        self.fc = nn.Linear(embed_size, output_size)
        self.encodings = InputEncoding(embed_size, vocab_size, device)
        self.decoder_blocks = nn.ModuleList(
            [
                DecoderBlock(
                    embed_size=embed_size,
                    num_heads=num_heads,
                    forward_expansion=forward_expansion,
                )
                for _ in range(num_decoders)
            ]
        )

    def forward(self, embeddings: Tensor, encoder_outs: Tensor, mask: Tensor) -> Tensor:
        decoder_block_outputs = self.encodings(embeddings)
        for decoder_block in self.decoder_blocks:
            decoder_block_outputs = decoder_block(
                decoder_block_outputs, encoder_outs, mask
            )
        outs = self.fc(decoder_block_outputs)

        return outs
