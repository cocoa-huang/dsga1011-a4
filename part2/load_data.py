import os, random, re, string
from collections import Counter
from tqdm import tqdm
import pickle

from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

import nltk
nltk.download('punkt')
from transformers import T5TokenizerFast
import torch

PAD_IDX = 0
PREFIX = "translate English to SQL: "
LT = "&lt;"
DECODER_START = PAD_IDX
TOKENIZER = T5TokenizerFast.from_pretrained("google-t5/t5-small")

class T5Dataset(Dataset):

    def __init__(self, data_folder, split):
        '''
        Skeleton for the class for performing data processing for the T5 model.

        Some tips for implementation:
            * You should be using the 'google-t5/t5-small' tokenizer checkpoint to tokenize both
              the encoder and decoder output. 
            * You want to provide the decoder some beginning of sentence token. Any extra-id on the
              T5Tokenizer should serve that purpose.
            * Class behavior should be different on the test set.
        '''
        self.split = split
        self.process_data(data_folder, split, TOKENIZER)

    def process_data(self, data_folder, split, tokenizer):
        nl = load_lines(os.path.join(data_folder, f"{split}.nl"))
        self.enc = [tokenizer(PREFIX + q).input_ids for q in nl]    
        if split != "test":
            sql = load_lines(os.path.join(data_folder, f"{split}.sql"))
            self.dec = [tokenizer(s.replace("<", LT)).input_ids for s in sql]

    
    def __len__(self):
        return len(self.enc)

    def __getitem__(self, idx):
        if self.split == "test":
            return self.enc[idx]
        return self.enc[idx], self.dec[idx]

def normal_collate_fn(batch):
    '''
    Collation function to perform dynamic padding for training and evaluation with the
    development or validation set.

    Inputs:
        * batch (List[Any]): batch is a list of length batch_size, where each index contains what
                             the dataset __getitem__ function returns.

    Returns: To be compatible with the provided training loop, you should be returning
        * encoder_ids: The input ids of shape BxT to be fed into the T5 encoder.
        * encoder_mask: Mask of shape BxT associated with padding tokens in the encoder input
        * decoder_inputs: Decoder input ids of shape BxT' to be fed into T5 decoder.
        * decoder_targets: The target tokens with which to train the decoder (the tokens following each decoder input)
        * initial_decoder_inputs: The very first input token to be decoder (only to be used in evaluation)
    '''
    enc = [torch.tensor(e) for e, _ in batch]
    dec = [torch.tensor(d) for _, d in batch]
    encoder_ids = pad_sequence(enc, batch_first=True, padding_value=PAD_IDX)
    encoder_mask = (encoder_ids != PAD_IDX).long()
    decoder_targets = pad_sequence(dec, batch_first=True, padding_value=PAD_IDX)
    start = torch.full((len(batch), 1), DECODER_START)
    decoder_inputs = torch.cat([start, decoder_targets[:, :-1]], dim=1)   # shift right by one
    return encoder_ids, encoder_mask, decoder_inputs, decoder_targets, start

def test_collate_fn(batch):
    '''
    Collation function to perform dynamic padding for inference on the test set.

    Inputs:
        * batch (List[Any]): batch is a list of length batch_size, where each index contains what
                             the dataset __getitem__ function returns.

    Recommended returns: 
        * encoder_ids: The input ids of shape BxT to be fed into the T5 encoder.
        * encoder_mask: Mask of shape BxT associated with padding tokens in the encoder input
        * initial_decoder_inputs: The very first input token to be decoder (only to be used in evaluation)
    '''
    enc = [torch.tensor(e) for e in batch]
    encoder_ids = pad_sequence(enc, batch_first=True, padding_value=PAD_IDX)
    encoder_mask = (encoder_ids != PAD_IDX).long()
    return encoder_ids, encoder_mask, torch.full((len(batch), 1), DECODER_START)

def get_dataloader(batch_size, split):
    data_folder = 'data'
    dset = T5Dataset(data_folder, split)
    shuffle = split == "train"
    collate_fn = normal_collate_fn if split != "test" else test_collate_fn

    dataloader = DataLoader(dset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)
    return dataloader

def load_t5_data(batch_size, test_batch_size):
    train_loader = get_dataloader(batch_size, "train")
    dev_loader = get_dataloader(test_batch_size, "dev")
    test_loader = get_dataloader(test_batch_size, "test")
    
    return train_loader, dev_loader, test_loader


def load_lines(path):
    with open(path, 'r') as f:
        lines = f.readlines()
        lines = [line.strip() for line in lines]
    return lines

def load_prompting_data(data_folder):
    # TODO
    return train_x, train_y, dev_x, dev_y, test_x


if __name__ == "__main__":
    enc, mask, dec_in, dec_tgt, start = next(iter(get_dataloader(4, "dev")))
    print(enc.shape, mask.shape, dec_in.shape, dec_tgt.shape, start.shape)
    assert dec_in.shape == dec_tgt.shape and (dec_in[:, 1:] == dec_tgt[:, :-1]).all()   # shift check
    print(TOKENIZER.decode(enc[0]), "\n", TOKENIZER.decode(dec_tgt[0][:30]))
