import torch
import json
from torchtext.data.metrics import bleu_score
from torchtext.vocab import Vocab
from model.transformer import Transformer
from collections.abc import Callable
from torch import Tensor
from typing import Union, List
import os


def save_model(model: Transformer, save_path: str):
    torch.save(model, save_path)


def save_as_json(arg: dict, save_path: str):
    with open(save_path, "w") as fp:
        json.dump(arg, fp)


def translate(
    model: Transformer,
    src_vocab: Vocab,
    tgt_vocab: Vocab,
    inputs: Union[str, List[str]],
    pipeline: Callable,
    max_tokens: int,
    start_token: int,
    end_token: int,
    device: str,
) -> List[str]:
    with torch.no_grad():
        inputs = pipeline(inputs, src_vocab)
        inputs = inputs.to(device)

        batch_size = inputs.shape[1]
        x_mask = model.src_mask(inputs)
        encoder_states = model.encoder(inputs, x_mask)
        output_tokens = []
        i = 0
        for i in range(inputs.shape[1]):
            y = torch.LongTensor([start_token]).reshape(-1, 1).to(device)

            current_output = []
            k = 0
            while True:
                predictions = model.decoder(y, encoder_states, None)

                predictions = predictions[-1, :, :].argmax(-1).unsqueeze(0)

                pred_tokens = tgt_vocab.lookup_token(predictions[-1].item())

                current_output.append(pred_tokens)
                y = torch.cat((y, predictions), dim=0)
                if end_token == predictions or len(current_output) >= max_tokens:
                    break
                k += 1
            output_tokens.append(" ".join(current_output))
    return output_tokens


def display_info(
    model: Transformer,
    loss: dict,
    epoch: int,
    bleu: dict,
    max_seqence: int,
    device: str,
    src_vocab: Vocab,
    tgt_vocab: Vocab,
    text_pipeline: Callable,
):
    print(f"########## epoch {epoch} ##########")
    print(f"train loss : {loss['train'][-1]}\t val loss: {loss['val'][-1]}")
    print(
        f"train bleu score : {bleu['train'][-1]} \t val bleu score : {bleu['val'][-1]}"
    )
    test_txt = "In this story an old man sets out to ask an Indian king to dig some well in his village when their water runs dry"
    expected_translation = "Dans cette histoire, un vieil homme entreprend de demander à un roi indien de creuser un puits dans son village lorsque leur eau sera à sec."
    predicted_translation = translate(
        model,
        src_vocab,
        tgt_vocab,
        test_txt,
        text_pipeline,
        max_seqence,
        tgt_vocab["<sos>"],
        tgt_vocab["<eos>"],
        device,
    )
    print(f"expected translation : {expected_translation}")
    print(f"predicted trainslation : {predicted_translation[0]}")
    print(
        f"bleu_score : {bleu_score([predicted_translation[0].split()],[[expected_translation.split()]],max_n=2,weights=[0.5,0.5])}"
    )
    print("\n-----------------------------------\n")


def get_bleu_score(tgt_vocab: Vocab, preds: Tensor, target: Tensor) -> float:
    preds = preds.detach()[1:].t().tolist()
    target = target.detach()[1:].t().tolist()
    pred_tokens = [tgt_vocab.lookup_tokens(tokens) for tokens in preds]
    target_tokens = [[tgt_vocab.lookup_tokens(tokens)] for tokens in target]
    bleu = bleu_score(pred_tokens, target_tokens, max_n=2, weights=[0.5, 0.5])
    return bleu
