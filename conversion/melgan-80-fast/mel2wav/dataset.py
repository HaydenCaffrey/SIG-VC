import torch
import torch.utils.data
import torch.nn.functional as F

from librosa.core import load
from librosa.util import normalize
from collections import namedtuple

from pathlib import Path
import numpy as np
import random
import os
import random
import re 

def files_to_list(filename):
    """
    Takes a text file of filenames and makes a list of filenames
    """
    with open(filename, encoding="utf-8") as f:
        files = f.readlines()

    files = [f.rstrip() for f in files]
    return files


class AudioDataset(torch.utils.data.Dataset):
    """
    This is the main class that calculates the spectrogram and returns the
    spectrogram, audio pair.
    """

    def __init__(self, training_files, segment_length, sampling_rate, augment=True):
        print(f'audio dataset : {(training_files, segment_length, sampling_rate, augment)}')
        self.sampling_rate = sampling_rate
        self.segment_length = segment_length
        self.audio_files = files_to_list(training_files)
        self.audio_files = [Path(training_files).parent / x for x in self.audio_files]
        random.seed(1234)
        random.shuffle(self.audio_files)
        self.augment = augment


    def __getitem__(self, index):
        # Read audio
        filename = self.audio_files[index]
        audio, sampling_rate = self.load_wav_to_torch(filename)
        # Take segment
        if audio.size(0) >= self.segment_length:
            max_audio_start = audio.size(0) - self.segment_length
            audio_start = random.randint(0, max_audio_start)
            audio = audio[audio_start : audio_start + self.segment_length]
        else:
            audio = F.pad(
                audio, (0, self.segment_length - audio.size(0)), "constant"
            ).data

        # audio = audio / 32768.0
        return audio.unsqueeze(0)

    def __len__(self):
        return len(self.audio_files)

    def load_wav_to_torch(self, full_path):
        """
        Loads wavdata into torch array
        """
        data, sampling_rate = load(full_path, sr=self.sampling_rate)
        data = 0.95 * normalize(data)

        if self.augment:
            amplitude = np.random.uniform(low=0.3, high=1.0)
            data = data * amplitude

        return torch.from_numpy(data).float(), sampling_rate



parallel_metaline = namedtuple('parallel_metaline', ['audio', 'mel', 'gta'])
SSB_BASE = '/NASdata/AudioData/AISHELL-ASR-SSB/SPEECHDATA/'
AUDIO_PATTERN  = re.compile(r'audio-(?P<sid>[SB0-9]+)/.npy')

class ParallelAudioDataset(torch.utils.data.Dataset):
    """
    This is the main class that loads and returns the spectrogram, audio pair.
    used to load gta spectrogram from a synthesizer to finetune the vocoder
    :param metafile: tacotron gta inference result mapping, which contains absolute audiofile path and its corresponding gta-mel file path
    :param segment_length: padding target-length in audio-time domain
    :param sampling_rate: is the sampling rate of input audio
    :param perturb: defaults to True, randomly shuffle metadata sequence
    """

    def __init__(
        self, 
        metafile, 
        segment_length, 
        sampling_rate, 
        perturb = True, 
        use_ground_truth_audio = False,
        use_numpy_audio = True
    ):
        print(f'audio dataset : {(metafile, segment_length, sampling_rate)}')
        self.sampling_rate = sampling_rate
        self.segment_length = segment_length
        self.metas = self.parse_metafile(metafile)

        self.use_numpy_audio = use_numpy_audio

        if perturb : 
            random.seed(1234)
            random.shuffle(self.metas)

        if use_ground_truth_audio : 
            self.audio_files = [Path(self.extract_original_audio_path(x.audio)) for x in self.metas]
        else : 
            self.audio_files = [Path(x.audio) for x in self.metas]  # list of audio-file paths
        self.mel_files   = [Path(x.gta) for x in self.metas]    # list of gta-mel paths

        self.stft_ratio = 200   # stft * len(s_t) == len(x_t)
        self.mel_segment_length = self.segment_length // self.stft_ratio
        assert self.segment_length % self.stft_ratio == 0, f'stft_ratio setting is wrong : {self.stft_ratio}'

    def extract_original_audio_path(self, audio_path) : 
        filename = os.path.basename(audio_path)
        m = AUDIO_PATTERN.match(filename)
        if m is not None : 
            sid = m.group('sid')
            spkid = sid[:7]
            return os.path.join(SSB_BASE, spkid, f'{sid}.wav')
        else : 
            raise AssertionError


    def parse_metafile(self, metafile_name) : 
        with open(metafile_name) as f : 
            return [parallel_metaline(*(line.strip().split('|')[:3])) for line in f] 

    def __getitem__(self, index):
        # Read audio
        audio = self.load_wav_to_torch(self.audio_files[index]) # (seqlen)
        mel   = torch.from_numpy(np.load(self.mel_files[index]))    # (mel_seqlen, mel_channel)
        mel = mel.transpose(0,1)
        #print(audio.shape, mel.shape)
        # audio, sampling_rate = self.load_wav_to_torch(filename)
        # Take segment
        max_mel_start = 0
        mel_start = 0
        if audio.size(0) >= self.segment_length:
            max_mel_start = mel.size(0) - self.mel_segment_length - 1
            #print(max_mel_start)
            mel_start = random.randint(0, max_mel_start)
            audio_start = mel_start * self.stft_ratio
            audio = audio[audio_start : audio_start + self.segment_length]
            mel   = mel[mel_start : mel_start + self.mel_segment_length]
        else:
            audio = F.pad(
                audio, (0, self.segment_length - audio.size(0)), "constant"
            )

            # mel is of shape (time,mel), we pad along the time axis, thus the excessive zeros
            mel = F.pad(
                mel, (0,0,0,self.mel_segment_length - mel.size(0))
            )

        # audio = audio / 32768.0
        if audio.size(0) != self.segment_length : 
            print(f'assertion failed, switch to all padding ({max_mel_start}/{mel_start} in {audio.size()})') 
            audio = torch.zeros((self.segment_length), dtype=audio.dtype)
            mel   = torch.zeros((self.mel_segment_length, 80))
        # print(f'audio shape : {audio.size()}, mel shape : {mel.size()}')
        return audio.unsqueeze(0), mel.permute((1,0))   # (1, time_x)  (mel, time_s)
        ## NOTE : debug this, not sure why does these `unsqueeze` calls appears here, maybe for compatibility reasons on the collation side ? 

    def __len__(self):
        return len(self.audio_files)

    def load_wav_to_torch(self, full_path):
        """
        Loads wavdata into torch array
        """

        if not self.use_numpy_audio : 
            data, _ = load(full_path, sr=self.sampling_rate)
        else : 
            data = np.load(full_path)
        data = 0.95 * normalize(data)

        # if self.augment:
            # amplitude = np.random.uniform(low=0.3, high=1.0)
            # data = data * amplitude

        return torch.from_numpy(data).float()
