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
model_sv = getattr(models_spk, 'ECAPA_TDNN')(80, hidden_dim=1024, embedding_size=256).cuda()


# print(model_sv.state_dict()['conv1.weight'][0,0:5])

checkpoint_sv = torch.load('/Netdata/2017/qinxy/ASV/DeepSpeaker/egs/SdSV/exp/vox2_80mel_ECAPA-TDNN_ASP_256_ArcFace-32-0.2_vc_ch_en_mix/model_96.pkl')
model_sv.load_state_dict(checkpoint_sv['model'])
# print(model_sv.state_dict()['module.conv1.weight'][0,0:5])
model_sv = nn.DataParallel(model_sv)
model_sv.eval()
with torch.no_grad():
    with open('/Netdata/zhanghz/generate_sv_data/wav/list.txt','r') as f:
        for line in f.readlines():
            wl = line.strip().replace('.wav','.npy')
            mel_dir = '/Netdata/zhanghz/generate_sv_data/mels/'
            embed_dir = '/Netdata/zhanghz/generate_sv_data/embeds/'
            os.makedirs(embed_dir, exist_ok=True)

            tar_mel = np.load(os.path.join(mel_dir, 'mel-'+wl))
            tar_mel = torch.from_numpy(tar_mel).cuda().float().to(device)
            tar_mel = tar_mel.unsqueeze(dim=0)
            #print(tar_mel.shape)
            embed = model_sv(tar_mel).cpu().detach().numpy()
            embed = np.squeeze(embed, 0)
            #print(embed.shape)
            np.save(os.path.join(embed_dir, 'embed'+wl), embed)
