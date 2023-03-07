import re
from pathlib import Path
import glob
import matplotlib.pyplot as plt
import numpy as np 
from tqdm import tqdm
from IPython.display import HTML, Audio, display
import random
import os
import sys
import itertools
import librosa
import soundfile
import time
import yaml
os.environ['CUDA_VISIBLE_DEVICES'] = '5'
import torch
from mel2wav import MelVocoder
import sys
#ssb_tacotron_130 finetune_cross_lingual
melgan = MelVocoder('logs/finetune_cross_lingual/',use_best = False, github=False)

from scipy import signal
emp = 0.99

class H() : 
    max_abs_value = 4.0
    min_level_db = -100

hparams = H()

def inv_preemphasis(wav, k, inv_preemphasize=True):
    if inv_preemphasize:
        return signal.lfilter([1], [1, -k], wav)
    return wav

def preemphasis(wav, k, preemphasize=True):
    if preemphasize:
        # wav = signal.lfilter([1, -0.75], [1], wav)
        return signal.lfilter([1, -k], [1], wav)
    return wav

def denormalize(D) : 
    return (((D + hparams.max_abs_value) * -hparams.min_level_db / (2 * hparams.max_abs_value))
                + hparams.min_level_db)

def _db_to_amp(x):
    return np.power(10.0, (x) * 0.05)


# reconstruction and display

def recon_from_mel(load_mel, use_tacotron_process = False) : 
    print(load_mel.dtype)
    if use_tacotron_process : 
        load_mel = denormalize(load_mel) 
        load_mel = (load_mel + 20) / 20
        load_mel = _db_to_amp(load_mel + 20)
    print(load_mel.dtype)
    mel_tensor = torch.from_numpy(load_mel.T)[None]
    recon = melgan.inverse(mel_tensor).squeeze().cpu().numpy()
    print(f'mel_tensor : ({mel_tensor.size()}) -> recon_wav : ({recon.shape})')
    
    if use_tacotron_process : 
        recon = inv_preemphasis(recon, 0.90)

    return recon.astype(np.float32)

def display_waveform(wav) : 
    mel = melgan(torch.from_numpy(wav)[None])[0].detach().cpu().numpy()
    print(f'audio sample ({wav.shape}, {mel.shape}) : ')
    display(Audio(wav,rate=16000))

    fig, axs = plt.subplots(1,2, figsize=(20,3))
    axs[0].plot(wav)
    axs[1].imshow(mel, aspect='auto', cmap='coolwarm')
    
def write_waveform(wav, filename) : 
#     librosa.output.write_wav(filename, wav, 16000)
    soundfile.write(filename, wav, 16000, 'PCM_16')

'''    
path='/Netdata/shiyao/tf_multispeaker/yyg_synth_data/trial_1/yyg/train-young.txt'
with open(path,'r') as p:
    text = p.readlines()
sp = np.load('/Netdata/shiyao/tf_multispeaker/yyg_synth_data/trial_1/yyg/path.npy', allow_pickle=True).item()
''path ='/Netdata/yangyg/ppg-vc/ppg-miya/logs-miya-ppg/out-set/eval/'
'''
#path ='/Netdata/yangyg/fastvoice/log-one-pretrained/syn_mel/'
import sys
path=sys.argv[1]
outpath = sys.argv[2]
os.makedirs(outpath, exist_ok=True)
for file in os.listdir(path):
#for line in text[:100000]:
    #wl = line.strip().split('|')
    #wav_fpath = out_dir.joinpath(file.replace('.npy','.wav'))
    #if wav_fpath.exists():
    #    print(wav_fpath)
    #    continue
    t = os.path.join(path,file)
    a = np.load(t, allow_pickle=True).squeeze().T
    if a.shape[1]!=80:
        a=a.T
    a = a.astype(np.float32)
    print(a.shape)

    wav=recon_from_mel((a))
    write_waveform(wav, os.path.join(outpath, file.replace('.npy','.wav')))
