import torch
from torchtext.data.utils import get_tokenizer
from torchtext.data import to_map_style_dataset
from torchtext.datasets import WikiText103,WikiText2
from torchtext.vocab import build_vocab_from_iterator
from torch.utils.data import DataLoader
from functools import partial

from utils.constants import (
MAX_SEQUENCE_LENGTH,
MIN_WORD_FREQUENCY,
CBOW_N_WORDS,
SKIPGRAM_N_WORDS)


def get_english_tokenizer():
    tokenizer=get_tokenizer(tokenizer="spacy",language="en_core_web_sm")
    return tokenizer

def get_data_iterator(dataset_name,data_dir,split_type):
    if dataset_name.lower()=="wikitext2":
        data_iterator=WikiText2(root=data_dir,split=(split_type))
    elif dataset_name.lower()=="wikitext103":
        data_iterator=WikiText103(root=data_dir,split=(split_type))
    else:
        raise ValueError("choose dataset from : WikiText2,WikiText103")
    data_iterator=to_map_style_dataset(data_iterator)
    return data_iterator

def build_vocab(tokenizer,data_iterator):
    vocab=build_vocab_from_iterator(
        map(tokenizer,data_iterator),
        min_freq=MIN_WORD_FREQUENCY,
        specials=["<unk>"]
    )
    vocab.set_default_index(vocab['<unk>'])
    return vocab

def collate_cbow(batch,text_pipeline):
    batch_input=[]
    batch_output=[]
    for text in batch:
        token_ids=text_pipeline(text)
        if len(token_ids)< CBOW_N_WORDS*2+1:
            continue
        if len(token_ids)>MAX_SEQUENCE_LENGTH:
            token_ids=token_ids[:MAX_SEQUENCE_LENGTH]
        for token_id in range(len(token_ids)-CBOW_N_WORDS*2):
           token_id_sequence=token_ids[token_id:token_id+CBOW_N_WORDS*2+1]
           output_token_id=token_id_sequence[CBOW_N_WORDS]
           input_token_ids=token_id_sequence
           batch_input.append(input_token_ids)
           batch_output.append(output_token_id)
    batch_input=torch.tensor(batch_input,dtype=torch.long)
    batch_output=torch.tensor(batch_output,dtype=torch.long)
    return batch_input,batch_output
def collate_skip(batch,text_pipeline):
    batch_input=[]
    batch_output=[]
    for text in batch:
        token_ids=text_pipeline(text)
        if len(token_ids)<SKIPGRAM_N_WORDS*2+1:
            continue
        if len(token_ids)>MAX_SEQUENCE_LENGTH:
            token_ids=token_ids[:MAX_SEQUENCE_LENGTH]
        for token_id in range(len(token_ids)-SKIPGRAM_N_WORDS*2):
            token_id_sequence=token_ids[token_id:token_id+SKIPGRAM_N_WORDS*2+1]
            input_token_id=token_id_sequence.pop(SKIPGRAM_N_WORDS)
            output_token_ids=token_id_sequence
            for output_token_id in output_token_ids:
                batch_input.append(input_token_id)
                batch_output.append(output_token_id)
    batch_input=torch.tensor(batch_input,dtype=torch.int32)
    batch_output=torch.tensor(batch_output,dtype=torch.int32)
    return batch_input,batch_output

def get_dataloader_and_vocab(model_name,dataset_name,split_type,data_dir,batch_size,shuffle,vocab=None):
    data_iter=get_data_iterator(dataset_name,data_dir,split_type)
    tokenizer=get_english_tokenizer()
    
    if not vocab:
        vocab=build_vocab(tokenizer,data_iter)
    
    text_pipeline=lambda x: vocab(tokenizer(x))
    if model_name=="cbow":
        collate_fn=collate_cbow
    elif model_name=="skipgram":
        collate_fn=collate_skip
    else:
        raise ValueError("select model from : cbow and skipgram")
    dataloader=DataLoader(
        data_iter,
        batch_size,
       shuffle,
       collate_fn=partial(collate_fn,text_pipeline=text_pipeline)
    )
    
    return dataloader, vocab
