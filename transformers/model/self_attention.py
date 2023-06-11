import torch
import torch.nn as nn
import math
from torch import Tensor

class MultiHeadAttention(nn.Module):
    def __init__(self,embed_size:int,num_heads:int) -> None:
        super().__init__()
        self.num_heads=num_heads
        self.head_len=embed_size//num_heads
        if self.head_len*num_heads != embed_size:
            raise ValueError("Embedding size must be divisible by heads")
        self.query=nn.Linear(embed_size,embed_size)
        self.key=nn.Linear(embed_size,embed_size)
        self.value=nn.Linear(embed_size,embed_size)
    def forward(self,queries:Tensor,keys:Tensor,values:Tensor,mask:Tensor=None)->Tensor:
        #shape=[seq_len,batch_size,embed_size]
        queries=self.query(queries)
        
        #shape=[seq_len,batch_size,embed_size]
        keys=self.key(keys)
        
        #shape=[seq_len,batch_size,embed_size]
        values=self.value(values)
        
        _,n_q,_=queries.shape
        queries=queries.reshape(n_q,-1,self.num_heads,self.head_len)
        _,n_k,_=keys.shape
        keys=keys.reshape(n_k,-1,self.num_heads,self.head_len)
        
        #batch_size(n),heads(h),query_seq_len(q),key_seq_len(l)
        attention_score=torch.einsum("nqhd,nkhd->nhqk",[queries,keys])
        scaled_attention=attention_score/math.sqrt(self.head_len)
        if mask is not None:
            scaled_attention=scaled_attention.masked_fill(mask==0, value=0)
        softmax_attention=torch.softmax(scaled_attention,-1)
        _,n_v,_=values.shape
        
        #batch_size(n),value_seq_len(l),num_heads(h),head_len(d)
        values=values.reshape(n_v,-1,self.num_heads,self.head_len)
        
        
        attention_value_filter=torch.einsum("nhql,nlhd->nqhd",[softmax_attention,values])
        n_a,_,_,_=attention_value_filter.shape
        
        #seq_len,batch_size,embed_size
        attention_value_filter=attention_value_filter.reshape([-1,n_a,self.num_heads*self.head_len])
        
        
        return attention_value_filter