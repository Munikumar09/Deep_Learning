import torch
from tqdm import tqdm
import os
from torchtext.data.metrics import bleu_score
import json
from torch.utils.tensorboard import SummaryWriter

class Trainer:
    def __init__(
        self,
        model,
        num_epochs,
        batch_size,
        criterion,
        optimizer,
        device,
        print_stats,
        tgt_vocab,
        src_vocab,
        text_pipeline,
    ):
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.print_stats=print_stats
        self.tgt_vocab=tgt_vocab
        self.src_vocab=src_vocab
        self.text_pipeline=text_pipeline
        self.loss={"train":[],'val':[]}
        self.bleu={"train":[],'val':[]}
        self.model=model.to(device)
        self.writer=SummaryWriter()
    def train(self, train_loader, val_loader, save_path):
        for epoch in range(self.num_epochs):
            self._train_epoch(train_loader)
            self._val_epoch(val_loader)
            if self.print_stats:
                print(f"########## epoch {epoch} ##########")
                print(f"train loss : {self.loss['train'][-1]}\t val loss: {self.loss['val'][-1]}")
                print(f"train bleu score : {self.bleu['train'][-1]} \t val bleu score : {self.bleu['val'][-1]}")
                test_txt="In this story an old man sets out to ask an Indian king to dig some well in his village when their water runs dry"
                expected_translation="Dans cette histoire, un vieil homme entreprend de demander à un roi indien de creuser un puits dans son village lorsque leur eau sera à sec."
                predicted_translation=self.transcribe(test_txt,self.text_pipeline,50,189,self.tgt_vocab,188)
                print(f"expected translation : {expected_translation}")
                print(f"predicted trainslation : {predicted_translation[0]}")
                print(f"bleu_score : {bleu_score([predicted_translation[0].split()],[[expected_translation.split()]],max_n=2,weights=[0.5,0.5])}")
                print("\n-----------------------------------\n")
            self.writer.add_scalar("train loss",scalar_value=self.loss['train'][-1],global_step=epoch)
            self.writer.add_scalar("train bleu",scalar_value=self.loss['train'][-1],global_step=epoch)
            self.writer.add_scalar("val loss",scalar_value=self.loss['val'][-1],global_step=epoch)
            self.writer.add_scalar("val bleu",scalar_value=self.bleu['val'][-1],global_step=epoch)
        self.save_model(os.path.join(save_path,"model.pt"))
        self.save_loss(os.path.join(save_path,"loss.json"))
        self.save_bleu(os.path.join(save_path,"bleu.json"))
    def _train_epoch(self,train_loader):
        i=0
        epoch_loss=0
        epoch_bleu=0
        for batch_src,batch_tgt in tqdm(train_loader):
            i+=1
            #moving the src batch and target batch into cuda
            source=batch_src.to(self.device)
            target=batch_tgt.to(self.device)
            #calling the model with inputs
            outputs=self.model(source,target)
            
            #predicted tokens [seq_len, batch_size]
            preds=outputs.argmax(-1)
            
            #reshaping from [seq_len,batch_size,vocab_size] to [seq_len*batch_size,vocab_size]
            #ignore the first seq because it is the start sentence
            outputs=outputs[1:].reshape(-1,outputs.shape[-1])
            
            #reshaping from [seq_len,batch_size] to [seq_len*batch_size]
            target_reshape=target[1:].reshape(-1)
            
            
            loss=self.criterion(outputs,target_reshape)
            epoch_loss+=loss.detach().item()
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # converting the indices to the str tokens to calculate bleu score
            preds=preds.detach()[1:].t().tolist()
            target=target.detach()[1:].t().tolist()
            pred_tokens=[self.tgt_vocab.lookup_tokens(tokens) for tokens in preds]
            target_tokens=[[self.tgt_vocab.lookup_tokens(tokens)] for tokens in target]
            bleu=bleu_score(pred_tokens,target_tokens,max_n=2,weights=[0.5,0.5])
            epoch_bleu+=bleu
            
        self.loss['train'].append(epoch_loss/len(train_loader))
        self.bleu['train'].append(epoch_bleu/len(train_loader))
                
            
    def _val_epoch(self,val_loader):
        i=0
        epoch_bleu=0
        epoch_loss=0
        with torch.no_grad():
            for source,target in tqdm(val_loader):
                i+=1
                source=source.to(self.device)
                target=target.to(self.device)
                outputs=self.model(source,target)
                preds=outputs.argmax(-1)
                outputs=outputs[1:].reshape(-1,outputs.shape[-1])
                target_reshape=target[1:].reshape(-1)
                loss=self.criterion(outputs,target_reshape)
                
                preds=preds.detach()[1:].t().tolist()
                target=target.detach()[1:].t().tolist()
                pred_tokens=[self.tgt_vocab.lookup_tokens(tokens) for tokens in preds]
                target_tokens=[[self.tgt_vocab.lookup_tokens(tokens)] for tokens in target]
                bleu=bleu_score(pred_tokens,target_tokens,max_n=2,weights=[0.5,0.5])
                epoch_bleu+=bleu
                epoch_loss+=loss.detach().item()
                
        self.loss['val'].append(epoch_loss/len(val_loader))
        self.bleu['val'].append(epoch_bleu/len(val_loader))
    def save_model(self,save_path):
        torch.save(self.model,save_path)
    def save_loss(self,save_path):
        with open(save_path,"w") as fp:
            json.dump(self.loss,fp)
    def save_bleu(self,save_path):
        with open(save_path,'w') as fp:
            json.dump(self.bleu,fp)
    def transcribe(self,inputs,pipeline,max_tokens,start_token,tgt_vocab,end_token):
        with torch.no_grad():
            inputs=pipeline(inputs,self.src_vocab)
            inputs=inputs.to(self.device)
            
            batch_size=inputs.shape[1]
            x_mask=self.model.src_mask(inputs)
            encoder_states=self.model.encoder(inputs,x_mask)
            output_tokens=[]
            i=0
            for i in range(inputs.shape[1]):
                y=torch.LongTensor([start_token]).reshape(-1,1).to(self.device)
                
                current_output=[]
                k=0
                while True:
                    
                    predictions=self.model.decoder(y,encoder_states,None)
                    
                    predictions=predictions[-1,:,:].argmax(-1).unsqueeze(0)
                   
                    pred_tokens=tgt_vocab.lookup_token(predictions[-1].item())
                    
                    current_output.append(pred_tokens)
                    y=torch.cat((y,predictions),dim=0)
                    if end_token==predictions or len(current_output)>=max_tokens:
                        break
                    k+=1
                output_tokens.append(" ".join(current_output))
        return output_tokens