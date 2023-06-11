import torch
from tqdm import tqdm
import os
from utils.helper import save_as_json, save_model
from torch.utils.tensorboard import SummaryWriter
from model.transformer import Transformer
from typing import Any
from collections.abc import Callable
from torch.utils.data import DataLoader


class Trainer:
    def __init__(
        self,
        model: Transformer,
        num_epochs: int,
        batch_size: int,
        criterion: Any,
        optimizer: Any,
        device: str,
        print_stats: bool,
        display_info: Callable,
        bleu_score: Callable,
    ):
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.print_stats = print_stats
        self.display_info = display_info
        self.bleu_score = bleu_score

        self.loss = {"train": [], "val": []}
        self.bleu = {"train": [], "val": []}
        self.model = model.to(device)
        self.writer = SummaryWriter()

    def train(self, train_loader: DataLoader, val_loader: DataLoader, save_path: str):
        for epoch in range(self.num_epochs):
            self._train_epoch(train_loader)
            self._val_epoch(val_loader)
            if self.print_stats:
                self.display_info(
                    model=self.model,
                    loss=self.loss,
                    epoch=epoch,
                    bleu=self.bleu,
                    max_seqence=50,
                    device=self.device,
                )

            self.writer.add_scalar(
                "train loss", scalar_value=self.loss["train"][-1], global_step=epoch
            )
            self.writer.add_scalar(
                "train bleu", scalar_value=self.bleu["train"][-1], global_step=epoch
            )
            self.writer.add_scalar(
                "val loss", scalar_value=self.loss["val"][-1], global_step=epoch
            )
            self.writer.add_scalar(
                "val bleu", scalar_value=self.bleu["val"][-1], global_step=epoch
            )
        save_model(self.model.state_dict(), os.path.join(save_path, "model.pt"))
        save_as_json(self.loss, os.path.join(save_path, "loss.json"))
        save_as_json(self.bleu, os.path.join(save_path, "bleu.json"))

    def _train_epoch(self, train_loader: DataLoader):
        epoch_loss = 0
        epoch_bleu = 0
        for batch_src, batch_tgt in tqdm(train_loader):
            # moving the src batch and target batch into cuda
            source = batch_src.to(self.device)
            target = batch_tgt.to(self.device)
            # calling the model with inputs
            outputs = self.model(source, target)

            # predicted tokens [seq_len, batch_size]
            preds = outputs.argmax(-1)

            # reshaping from [seq_len,batch_size,vocab_size] to [seq_len*batch_size,vocab_size]
            # ignore the first seq because it is the start sentence
            outputs = outputs[1:].reshape(-1, outputs.shape[-1])

            # reshaping from [seq_len,batch_size] to [seq_len*batch_size]
            target_reshape = target[1:].reshape(-1)

            loss = self.criterion(outputs, target_reshape)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            epoch_loss += loss.item()

            # converting the indices to the str tokens to calculate bleu score
            bleu = self.bleu_score(preds, target)
            epoch_bleu += bleu

        self.loss["train"].append(epoch_loss / len(train_loader))
        self.bleu["train"].append(epoch_bleu / len(train_loader))

    def _val_epoch(self, val_loader: DataLoader):
        i = 0
        epoch_bleu = 0
        epoch_loss = 0
        with torch.no_grad():
            for source, target in tqdm(val_loader):
                i += 1
                source = source.to(self.device)
                target = target.to(self.device)
                outputs = self.model(source, target)
                preds = outputs.argmax(-1)
                outputs = outputs[1:].reshape(-1, outputs.shape[-1])
                target_reshape = target[1:].reshape(-1)
                loss = self.criterion(outputs, target_reshape)

                bleu = self.bleu_score(preds, target)
                epoch_bleu += bleu
                epoch_loss += loss.detach().item()

        self.loss["val"].append(epoch_loss / len(val_loader))
        self.bleu["val"].append(epoch_bleu / len(val_loader))
