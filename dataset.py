from utils import *
import pandas as pd
import torch
import torchaudio
import torchaudio.transforms as T
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizerFast
from transformers import Wav2Vec2Processor
import numpy as np

class MultiSpeech(Dataset):
    def __init__(self, audio, text, mask, a, o, l):

        n_fft = 512
        win_length = None
        hop_length = 256
        n_mels = 256
        sample_rate = 6000

        self.audio = audio
        self.text = text
        self.mask = mask
        self.a = a
        self.o = o
        self.l = l

        self.mel_spectrogram = Wav2Vec2Processor.from_pretrained("facebook/hubert-large-ls960-ft")

        self.class_dict_a, self.class_dict_o, self.class_dict_l = get_class_dicts()

    def __len__(self):
        return len(self.audio)

    def __getitem__(self, index):

        SPEECH_WAVEFORM, sr = torchaudio.load('task_data/'+ self.audio[index])

        melspec = self.mel_spectrogram(SPEECH_WAVEFORM,  sampling_rate=sr, return_tensors="pt").input_values[0][0]

        return melspec, self.text[index], self.mask[index], self.class_dict_a[self.a[index]], self.class_dict_o[self.o[index]], self.class_dict_l[self.l[index]]


        
def pad_sequence(batch):
    # Make all tensor in a batch the same length by padding with zeros
    batch = [item for item in batch]
    batch = torch.nn.utils.rnn.pad_sequence(batch, batch_first=True, padding_value=0.)
    return batch.permute(0, 1, 2)


def collate_fn(batch):

    # A data tuple has the form:
    # waveform, sample_rate, label, speaker_id, utterance_number

    tensors, text, mask, a, o, l = [], [], [], [], [], []

    # Gather in lists, and encode labels as indices
    for waveform, t, m, i, j , k in batch:

        text += [t]
        mask += [m]
        tensors += [waveform]
        a += torch.Tensor([i])
        o += torch.Tensor([j])
        l += torch.Tensor([k])
    # Group the list of tensors into a batched tensor
    tensors = pad_sequence(tensors)
    a = torch.stack(a)
    o = torch.stack(o)
    l = torch.stack(l)
    text = torch.stack(text)
    mask = torch.stack(mask)

    return tensors, text, mask, a.type(torch.LongTensor) , o.type(torch.LongTensor) , l.type(torch.LongTensor) 


def get_loader(path, max_len, batch_size, num_worker, shuffle):

    df = pd.read_csv(path)

    text = []
    for i in df['transcription']:
        text.append(clean_text(i))

    tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
    tokens = tokenizer.batch_encode_plus(
    text,
    max_length = max_len,
    pad_to_max_length=True,
    truncation=True,
    return_token_type_ids=False
)


    a_t = np.array(df['action'])
    o_t = np.array(df['object'])
    l_t = np.array(df['location'])
    audio_train = np.array(df['path'])
    train_seq = torch.tensor(tokens['input_ids'])
    train_mask = torch.tensor(tokens['attention_mask'])


    ds = MultiSpeech(audio_train, train_seq, train_mask, a_t, o_t, l_t)
    loader =  DataLoader(ds, batch_size=batch_size, collate_fn=collate_fn, shuffle=shuffle, num_workers=num_worker)

    return loader

