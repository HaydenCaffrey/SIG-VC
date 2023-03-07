import torch
from torch.utils.data import Dataset, DataLoader
import kaldiio
import numpy as np
import math
import os

import hparams
import audio as Audio
from utils import pad_1D, pad_2D, process_meta
from text import text_to_sequence, sequence_to_text

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Dataset(Dataset):
    def __init__(self, filename="../../train.txt", sort=True):
        self.basename  = process_meta(
            filename
        )
        self.sort = sort
        self._mel_dir = os.path.join(os.path.dirname(filename), 'mels')
    def __len__(self):
        return len(self.basename)

    def __getitem__(self, idx):
        meta = self.basename[idx]

        x = meta[-1].replace('.wav','.wav.npy')
        bnf = np.load('bnf/'+x, allow_pickle=True)

        mel_target = np.load(os.path.join(self._mel_dir, meta[1]))
        
        self._embed_dir=os.path.join(hparams.base_dir, hparams.embed_type)
        embedding = meta[2]
        embed = np.load(os.path.join(self._embed_dir, embedding))
        speaker = [0]
 
        sample = {
                  "bnf": bnf,
                  "mel_target": mel_target,
                  "embed": embed,
                  "speaker": speaker
                }

        return sample

    def reprocess(self, batch, cut_list):
        bnfs = [batch[ind]["bnf"] for ind in cut_list]
        mel_targets = [batch[ind]["mel_target"] for ind in cut_list]
        embeds = [batch[ind]["embed"] for ind in cut_list]
        speaker = [batch[ind]["speaker"] for ind in cut_list]

        length_text = np.array(list())
        for bnf in bnfs:
            length_text = np.append(length_text, bnf.shape[0])

        length_mel = np.array(list())
        for mel in mel_targets:
            length_mel = np.append(length_mel, mel.shape[0])

        embeds = np.array(embeds)
        bnfs = pad_2D(bnfs)
        mel_targets = pad_2D(mel_targets, PAD=-4)
        speaker = np.array(speaker)

        out ={"bnf": bnfs,
              "embed": embeds,
              "speaker": speaker,
              "mel_target": mel_targets,
               "src_len": length_text,
               "mel_len": length_mel}

        return out

    def collate_fn(self, batch):
        len_arr = np.array([d["bnf"].shape[0] for d in batch])
        index_arr = np.argsort(-len_arr)
        batchsize = len(batch)
        real_batchsize = int(math.sqrt(batchsize))

        cut_list = list()
        for i in range(real_batchsize):
            if self.sort:
                cut_list.append(
                    index_arr[i*real_batchsize:(i+1)*real_batchsize])
            else:
                cut_list.append(
                    np.arange(i*real_batchsize, (i+1)*real_batchsize))

        output = list()
        for i in range(real_batchsize):
            output.append(self.reprocess(batch, cut_list[i]))

        return output


if __name__ == "__main__":
    # Test
    dataset = Dataset('val.txt')
    training_loader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=dataset.collate_fn,
                                 drop_last=True, num_workers=0)
    total_step = hparams.epochs * len(training_loader) * hparams.batch_size

    cnt = 0
    for i, batchs in enumerate(training_loader):
        for j, data_of_batch in enumerate(batchs):
            mel_target = torch.from_numpy(
                data_of_batch["mel_target"]).float().to(device)
            D = torch.from_numpy(data_of_batch["D"]).int().to(device)
            if mel_target.shape[1] == D.sum().item():
                cnt += 1

    print(cnt, len(dataset))
