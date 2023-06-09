import torch.nn as nn
from model.decoder_block import DecoderBlock
from input_encoding import InputEncoding
class Decoder(nn.Module):
    def __init__(
        self, embed_size, num_heads,forward_expansion, num_decoders,output_size,vocab_size,device
    ) -> None:
        super().__init__()
        self.embed_size = embed_size
        self.device=device
        self.num_heads = num_heads
        self.forward_expansion = forward_expansion
        self.num_decoders = num_decoders
        self.fc=nn.Linear(embed_size,output_size)
        self.encodings=InputEncoding(embed_size,vocab_size,device)
        self.decoder_blocks=nn.ModuleList([DecoderBlock(
                embed_size=self.embed_size,
                num_heads=self.num_heads,
                forward_expansion=self.forward_expansion,
            ) for _ in range(self.num_decoders)])
    def forward(self, embeddings, encoder_outs,mask):
        decoder_block_outputs = self.encodings(embeddings)
        
        for decoder_block in self.decoder_blocks:
            decoder_block_outputs=decoder_block(decoder_block_outputs,encoder_outs,mask)
            
        outs=self.fc(decoder_block_outputs)
        return outs
        