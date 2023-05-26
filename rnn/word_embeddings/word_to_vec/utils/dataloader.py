from torch.utils.data import DataLoader
from functools import partial
import random
 
def get_context(int_words,idx,max_window_size):
    window_size=random.randint(1,max_window_size)
    start=max(0,idx-window_size)
    end=min(idx+window_size+1,len(int_words)-1)
    context_words=int_words[start:idx]+int_words[idx+1:end]
    return context_words

def collate_skip(batch,max_window_size):
    batch_target,batch_context=[],[]
    print(f"batch data : {batch}")
    for i in range(len(batch)):
        target_word=batch[i]
        context_words=get_context(batch,i,max_window_size)
        batch_target.extend([target_word]*len(context_words))
        batch_context.extend(context_words)
    return batch_target,batch_context

def get_dataloader(dataset,batch_size,shuffle,max_window_size):
  
    dataloader=DataLoader(
        dataset,
        batch_size,
        shuffle,
        collate_fn=partial(collate_skip,max_window_size)
    )
    return dataloader
