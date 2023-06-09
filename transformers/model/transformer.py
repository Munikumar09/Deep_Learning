import torch.nn as nn
class Transformer(nn.Module):
    def __init__(self,encoder,decoder) -> None:
        super().__init__()
        self.encoder=encoder
        self.decoder=decoder
    def forward(self,src_embeddings,tgt_embeddings):
        encoder_outputs=self.encoder(src_embeddings)
        decoder_outputs=self.decoder(tgt_embeddings,encoder_outputs)
        return decoder_outputs
