import torch.nn as nn
from utils.constants import EMBED_DIMENSION, EMBED_MAX_NORM
from torchsummary import summary
import torch

class CBOWModel(nn.Module):
    def __init__(self, vocab_size: int):
        super(CBOWModel, self).__init__()
        self.embedding = nn.Embedding(
            vocab_size, 
            EMBED_DIMENSION, 
            max_norm=EMBED_MAX_NORM
        )
        self.linear=nn.Linear(in_features=EMBED_DIMENSION,out_features=vocab_size)
    def forward(self,inputs):
        outputs=self.embedding(inputs)
        outputs=outputs.mean(axis=1)
        outputs=self.linear(outputs)
        return outputs
    
class NegativeSamplingLoss(nn.Module):
    def __init__(self):
        super(NegativeSamplingLoss, self).__init__()
    def forward(self,
            target_vector,
            context_vector,
            noise_vector):
        batch_size,embedding_size=target_vector.shape
        target_vector=target_vector.view(batch_size,embedding_size,1)
        context_vector=context_vector.view(batch_size,1,embedding_size)
        out_loss=torch.bmm(context_vector,target_vector).sigmoid().log().squeeze()
        noise_loss=torch.bmm(noise_vector.neg(),target_vector).sigmoid().log()
        noise_loss=noise_loss.squeeze().sum(1)
        return -(out_loss+noise_loss).mean()

class SkipGramNegSampling(nn.Module):
    def __init__(self,n_vocab,n_embed,noise_dist,device):
        super(SkipGramNegSampling,self).__init__()
        self.n_vocab=n_vocab
        self.n_embed=n_embed
        self.noise_dist=noise_dist
        self.device=device
        self.context_embed=nn.Embedding(n_vocab,n_embed,max_norm=1)
        self.target_embed=nn.Embedding(n_vocab,n_embed,max_norm=1)
    def forward_context(self,contexts):
        embed_contexts=self.context_embed(contexts)
        return embed_contexts
    
    def forward_target(self,target):
        embed_target=self.target_embed(target)
        return embed_target
    
    def forward_noise(self,batch_size,n_samples):
        if self.noise_dist is None:
            noise=torch.ones(self.n_vocab)
        else:
            noise=self.noise_dist
        
        noise_words=torch.multinomial(noise,
                              num_samples=batch_size*n_samples,
                              replacement=True)
        noise_words.to(self.device)
        
        noise_vector=self.target_embed(noise_words)

        noise_vector=noise_vector.view(batch_size,n_samples,self.n_embed)
        
        return noise_vector
