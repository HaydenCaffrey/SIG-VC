import torch
import torch.nn as nn
import numpy as np
import os
import argparse
import re
from string import punctuation

from fastspeech2 import FastSpeech2
import sv_model.modules.model_spk as models_spk
import hparams as hp
import utils
import audio as Audio


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model_sv = getattr(models_spk, hp.sv_model_type)(80, hidden_dim=1024, embedding_size=256).cuda()
# model_sv = getattr(models_spk, hp.sv_model_type)(hp.in_planes, hp.embd_dim, dropout=0).cuda()
# print(model_sv.state_dict()['front.conv1.weight'][0])
checkpoint_sv = torch.load('sv_model/save_model/%s/final.pkl' % hp.sv_model_name )
# checkpoint_sv = torch.load('/Netdata/2017/qinxy/ASV/DeepSpeaker/egs/SdSV/exp/vox2_80mel_ResNet50SEStatsPool-64-256_ArcFace-32-0.2_vc/model_39.pkl' )
model_sv.load_state_dict(checkpoint_sv['model']) #load state dict sv model
# print(model_sv.state_dict()['front.conv1.weight'][0])
model_sv  = nn.DataParallel(model_sv)
model_sv.eval() # sv model state
#return model
print('loading complete!')


import torch
import torch.nn as nn
import numpy as np
import os
import argparse
import re
from string import punctuation

import hparams as hp
import utils
import audio as Audio
from tqdm import tqdm

tar_embed_path = './embeds/'
os.makedirs(tar_embed_path, exist_ok = True)

mel_path = './mels/'

with torch.no_grad():
    ff = os.listdir(mel_path)
    for line in tqdm(ff):
        tar_embed_name = line.replace('mel','embed')
        mel = np.load(os.path.join(mel_path, line))
        mel = torch.from_numpy(mel).cuda().float().to(device)
        mel = mel.unsqueeze(dim=0)
        
        embed = model_sv(mel).cpu().detach().numpy()
        embed = np.squeeze(embed, 0)
        
        np.save(os.path.join(tar_embed_path, tar_embed_name),embed)

