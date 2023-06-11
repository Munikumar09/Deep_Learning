import torch.nn as nn
import torch
from model.transformer_encoder import Encoder
from model.transformer_decoder import Decoder
from torch import Tensor


class Transformer(nn.Module):
    def __init__(
        self,
        encoder: Encoder,
        decoder: Decoder,
        src_pad_idx: int,
        tgt_pad_idx: int,
        device: str,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_pad_idx = src_pad_idx
        self.tgt_pad_idx = tgt_pad_idx
        self.device = device


    def src_mask(self, src: Tensor) -> Tensor:
        # [seq_len,1,1,batch_size]
        mask = (src.t() != self.src_pad_idx).unsqueeze(1).unsqueeze(2)

        return mask.to(self.device)


    def tgt_mask(self, tgt: Tensor) -> Tensor:
        tgt_len, N = tgt.shape

        # [batch_size,1,seq_len,seq_len]
        tgt_mask = torch.tril(torch.ones((tgt_len, tgt_len))).expand(
            N, 1, tgt_len, tgt_len
        )

        return tgt_mask.to(self.device)


    def forward(self, src: Tensor, tgt: Tensor) -> Tensor:
        src_mask = self.src_mask(src)
        tgt_mask = self.tgt_mask(tgt)

        # [seq_len,batch_size,embed_size]
        encoder_outputs = self.encoder(src, src_mask)

        # [seq_len,batch_size,output_vocab_size]
        decoder_outputs = self.decoder(tgt, encoder_outputs, tgt_mask)

        return decoder_outputs
