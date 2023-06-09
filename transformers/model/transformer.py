import torch.nn as nn
import torch
class Transformer(nn.Module):
    def __init__(self,encoder,decoder,src_pad_idx,tgt_pad_idx,device) -> None:
        super().__init__()
        self.encoder=encoder
        self.decoder=decoder
        self.src_pad_idx=src_pad_idx
        self.tgt_pad_idx=tgt_pad_idx
        self.device=device
    def src_mask(self,src):
        mask=(src.t()!=self.src_pad_idx).unsqueeze(1).unsqueeze(2)
        return mask.to(self.device)
    def tgt_mask(self,tgt):
        tgt_len,N=tgt.shape
        tgt_mask=torch.tril(torch.ones((tgt_len,tgt_len))).expand(N,1,tgt_len,tgt_len)
        return tgt_mask.to(self.device)
    def forward(self,src,tgt):
        src_mask=self.src_mask(src)
        tgt_mask=self.tgt_mask(tgt)
        encoder_outputs=self.encoder(src,src_mask)
        decoder_outputs=self.decoder(tgt,encoder_outputs,tgt_mask)
        return decoder_outputs
