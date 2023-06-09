import torch
from tqdm import tqdm
import os
from torchtext.data.metrics import bleu_score
import json

class Trainer:
    def __init__(
        self,
        model,
        num_epochs,
        batch_size,
        criterion,
        optimizer,
        learning_rate,
        output_size_decoder,
        teacher_forcing_ratio,
        device,
        print_stats,
        tgt_vocab,
        pred_pipeline,
    ):
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.output_size_decoder=output_size_decoder
        self.criterion = criterion
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.teacher_forcing_ratio = teacher_forcing_ratio
        self.device = device
        self.print_stats=print_stats
        self.tgt_vocab=tgt_vocab
        self.pred_pipeline=pred_pipeline
        self.loss={"train":[],'val':[]}
        self.bleu={"train":[],'val':[]}
        self.model=model.to(device)
    def train(self, train_loader, val_loader, save_path):
        for epoch in tqdm(range(self.num_epochs)):
            self._train_epoch(train_loader)
            self._val_epoch(val_loader)
            if self.print_stats:
                print(f"train loss : {self.loss['train'][-1]}\t val loss: {self.loss['val'][-1]}")
                print(f"train bleu score : {self.bleu['train'][-1]} \t val bleu score : {self.bleu['val'][-1]}")
                test_txt="In this story an old man sets out to ask an Indian king to dig some well in his village when their water runs dry"
                expected_translation="Dans cette histoire, un vieil homme entreprend de demander à un roi indien de creuser un puits dans son village lorsque leur eau sera à sec."
                predicted_translation=self.transcribe(test_txt,self.pred_pipeline,50,189,self.tgt_vocab,188)
                print(f"expected translation : \t {expected_translation}")
                print(f"predicted trainslation : \t{predicted_translation}")
                print(f"bleu_score : {bleu_score([predicted_translation[0].split()],[[expected_translation.split()]],max_n=2,weights=[0.5,0.5])}")
                print("\n-----------------------------------\n")
        self.save_model(os.path.join(save_path,"model.pt"))
        self.save_loss(os.path.join(save_path,"loss.json"))
    def _train_epoch(self,train_loader):
        i=0
        epoch_loss=0
        epoch_bleu=0
        for batch_src,batch_tgt in train_loader:
            i+=1
            source=batch_src.to(self.device)
            target=batch_tgt.to(self.device)
            outputs=self.model(source,target,self.output_size_decoder,self.teacher_forcing_ratio)
            preds=outputs.argmax(-1)
            outputs=outputs[1:].reshape(-1,outputs.shape[-1])
            batch_tgt=batch_tgt[1:].reshape(-1)
            loss=self.criterion(outputs,batch_tgt)
            epoch_loss+=loss.item()
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            pred_tokens=[self.tgt_vocab.lookup_tokens(tokens) for tokens in preds[1:].t().tolist()]
            target_tokens=[[self.tgt_vocab.lookup_tokens(tokens)] for tokens in target.t().tolist()]
            bleu=bleu_score(pred_tokens,target_tokens,max_n=2,weights=[0.5,0.5])
            epoch_bleu+=bleu
            if i>=5:
                break
        self.loss['train'].append(epoch_loss/i)
        self.bleu['train'].append(epoch_bleu/i)
                
            
    def _val_epoch(self,val_loader):
        i=0
        epoch_bleu=0
        epoch_loss=0
        with torch.no_grad():
            for batch_src,batch_tgt in val_loader:
                i+=1
                source=batch_src.to(self.device)
                target=batch_tgt.to(self.device)
                outputs=self.model(source,target,self.output_size_decoder,self.teacher_forcing_ratio)
                preds=outputs.argmax(-1)
                outputs=outputs[1:].reshape(-1,outputs.shape[-1])
                batch_tgt=batch_tgt[1:].reshape(-1)
                loss=self.criterion(outputs,batch_tgt)
                pred_tokens=[self.tgt_vocab.lookup_tokens(tokens) for tokens in preds[1:].t().tolist()]
                target_tokens=[[self.tgt_vocab.lookup_tokens(tokens)] for tokens in target.t().tolist()]
                bleu=bleu_score(pred_tokens,target_tokens,max_n=2,weights=[0.5,0.5])
                epoch_bleu+=bleu
                epoch_loss+=loss.item()
                if i>=5:
                    break
        self.loss['val'].append(epoch_loss/i)
        self.bleu['val'].append(epoch_bleu/i)
    def save_model(self,save_path):
        torch.save(self.model,save_path)
    def save_loss(self,save_path):
        with open(save_path,"w") as fp:
            json.dump(self.loss,fp)
    
    def transcribe(self,inputs,pipeline,max_tokens,start_token,tgt_vocab,end_token):
        with torch.no_grad():
            inputs=pipeline(inputs)
            inputs=inputs.to(self.device)
            encoder_states,hidden,cell=self.model.encoder(inputs)
            output_tokens=[]
            i=0
            for i in range(inputs.shape[1]):
                x=torch.LongTensor([start_token]).to(self.device)
                current_output=[]
                while True:
                    predictions=self.model.decoder(x,hidden,cell)
                    predictions=predictions.argmax(1)
                    pred_tokens=tgt_vocab.lookup_token(predictions.item())
                    current_output.append(pred_tokens)
                    x=predictions
                    if end_token==predictions or len(current_output)>=max_tokens:
                        break
                output_tokens.append(" ".join(current_output))
        return output_tokens