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
import time
import yaml
os.environ['CUDA_VISIBLE_DEVICES'] = '6'
import torch
from mel2wav import MelVocoder
from mel2wav import load_model


melgan = MelVocoder('logs/ssb_tacotron_130/',use_best = False, github=False)
melgan_model = load_model('logs/ssb_tacotron_130/', use_best = False, device='cuda')

dummy_input = torch.randn(1,80, 200).cuda()
torch.onnx.export(melgan_model, dummy_input, 'melgan.onnx',  opset_version=10, do_constant_folding=True, input_names=["mel"], output_names=["audio"])

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
    librosa.output.write_wav(filename, wav, 16000)

'''    
path='/Netdata/shiyao/tf_multispeaker/yyg_synth_data/trial_1/yyg/train-young.txt'
with open(path,'r') as p:
    text = p.readlines()
sp = np.load('/Netdata/shiyao/tf_multispeaker/yyg_synth_data/trial_1/yyg/path.npy', allow_pickle=True).item()
''path ='/Netdata/yangyg/ppg-vc/ppg-miya/logs-miya-ppg/out-set/eval/'
'''
path ='/Netdata/yangyg/ppg-vc/bneck-vc/logs-gmm-ljs/vctk-12-25/eval/'
out_dir = Path("/Netdata/yangyg/feedback/logs-miya-feedback-weight-sum/").joinpath("sum", "syn_wav")
for file in os.listdir(path):
#for line in text[:100000]:
    #wl = line.strip().split('|')
    #wav_fpath = out_dir.joinpath(file.replace('.npy','.wav'))
    #if wav_fpath.exists():
    #    print(wav_fpath)
    #    continue
    t = os.path.join(path,file)
    a = np.load(t).squeeze().T
    a = a.astype(np.float32)
    print(a.shape)

    wav=recon_from_mel((a.T))
    write_waveform(wav, os.path.join('/Netdata/yangyg/ppg-vc/bneck-vc/logs-gmm-ljs/vctk-12-25/syn_wav/',file.replace('.npy','.wav')))
    break
