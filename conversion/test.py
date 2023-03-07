import torch
import torch.nn as nn
import numpy as np
import os
import argparse
import re
from string import punctuation
import sv_model.modules.model_spk as models_spk
from fastspeech2 import FastSpeech2
import hparams as hp
import utils
import audio as Audio


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
class Fastspeech_Synthesiszer() :
    def __init__(self, num):
#         hp.checkpoint_path = './aishell/ckpt/'
        checkpoint_path = os.path.join(
            hp.checkpoint_path, "checkpoint_{}.pth.tar".format(num))
        print(checkpoint_path)
        self.model = nn.DataParallel(FastSpeech2()).to(device)
        self.model.load_state_dict(torch.load(checkpoint_path)['model'])
        self.model.requires_grad = False
        self.model.eval()
        
        self.model_sv = getattr(models_spk, hp.sv_model_type)(80, hidden_dim=1024, embedding_size=256).cuda() #tdnn
        checkpoint_sv = torch.load('./sv_model/save_model/%s/model_94.pkl' % hp.sv_model_name ) #tdnn
        
        self.model_sv.load_state_dict(checkpoint_sv['model']) #load state dict sv model
        self.model_sv.requires_grad = False
        self.model_sv.eval()


    def synthesize(self, bnf, embed, embed_tar):

        bnf = np.array(bnf)
        bnf = np.stack([bnf])
        embed = np.array(embed)
        embed = np.stack([embed])
        embed_tar = np.array(embed_tar)
        embed_tar = np.stack([embed_tar])


        bnf = torch.from_numpy(bnf).cuda().float().to(device)
        src_len = torch.from_numpy(np.array([bnf.shape[1]])).to(device)
        embed = torch.from_numpy(embed).cuda().float().to(device)
        embed_tar = torch.from_numpy(embed_tar).cuda().float().to(device)
#         print(embed.shape, bnf.shape, src_len)


        with torch.no_grad():
            _, mel_mid, mel, mel_postnet, _, _ = self.model(bnf, src_len, mel_len=src_len, embeds=embed, embeds_tar=embed_tar)
        pre_embd = self.model_sv(mel_postnet)
        spk_loss = 1-torch.cosine_similarity(pre_embd, embed_tar, dim=-1)
        print(spk_loss.item())
        
        mel_postnet_torch = mel_postnet.transpose(1, 2).detach()
        mel_postnet = mel_postnet[0].cpu().transpose(0, 1).detach()
        
        mel_mid_torch = mel_mid.transpose(1, 2).detach()
        mel_mid = mel_mid[0].cpu().transpose(0, 1).detach()

        return mel_postnet, mel_mid

    
    
  
    
parser = argparse.ArgumentParser()
parser.add_argument('--test_step', type=int, default=0)
args = parser.parse_args()
synther = Fastspeech_Synthesiszer(args.test_step)  
print('loading complete!')



import kaldiio
import time
import hparams as hp
import os 


embed_path = "./embeds/"
mel_path = './mels/'
bnf_path = './bnf/'
T2_output_range = (-4, 4)
logdir = os.path.join(hp.dataset,'syn_mel/')
os.makedirs(logdir, exist_ok=True)

src_lst = os.listdir('./test_data/source/')
tar_lst = os.listdir('./test_data/target/')
for src in src_lst:
    for tar in tar_lst:
        x = src.replace('.wav','')
        bnf = np.load(os.path.join(mel_path, x+'.wav.npy'))
        mel = np.load(os.path.join(mel_path, 'mel-'+x+'.npy'))
        embed = np.load(os.path.join(embed_path, 'embed-'+x+'.npy'))

        T_b = bnf.shape[0]
        T_m = mel.shape[0]
        if T_b>T_m:
            bnf = bnf[:T_m,:]
        else:
            mel = mel[:T_b,:]

        y = tar.replace('.wav','')
        embed_tar = np.load(os.path.join(embed_path, 'embed-'+target+'.npy'))
        mel, mel_mid =synther.synthesize(bnf, embed, embed_tar)


        mel = mel.numpy()
        mels = [np.clip(tmp, T2_output_range[0], T2_output_range[1]) for tmp in mel]
        mel =np.array(mels)

        mel_mid = mel_mid.numpy()
        mels_mid = [np.clip(tmp, T2_output_range[0], T2_output_range[1]) for tmp in mel_mid]
        mel_mid =np.array(mels_mid)
        print(x, target)
        name =logdir+x+'--'+y
        np.save(name,mel)