from torch.utils.data import Dataset
from data.helper import build_vocab, preprocess, separate_src_tgt, train_test_split


class CustomDataset(Dataset):
    def __init__(self, dataset) -> None:
        super().__init__()
        self.src_data = dataset[0]
        self.tgt_data = dataset[1]

    def __len__(self):
        return len(self.src_data)

    def __getitem__(self, index):
        return self.src_data[index], self.tgt_data[index]


def data_process_pipeline(data, eng_vocab=None, fra_vocab=None):
    src, tgt = separate_src_tgt(data)
    if eng_vocab is None:
        eng_vocab = build_vocab(src)
        fra_vocab = build_vocab(tgt)
    src_idx = [eng_vocab.forward(sent) for sent in src]
    tgt_idx = [fra_vocab.forward(sent) for sent in tgt]
    train_dataset = CustomDataset((src_idx, tgt_idx))
    return train_dataset, eng_vocab, fra_vocab


def load_data(data_path, train_percent):
    with open(data_path, "r", encoding="utf-8") as fp:
        data = fp.read()
    clean_data = preprocess(data)
    sent_list = [sent for sent in clean_data.split("\n") if len(sent) > 0]
    sorted_sent_list = sorted(sent_list, key=lambda x: len(x.split("\t")[0].split(" ")))
    train_data, test_data = train_test_split(sorted_sent_list, train_percent)
    return train_data, test_data
