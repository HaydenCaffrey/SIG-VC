import torch
import torch.nn as nn
import hparams as hp
from classifier import ReversalClassifier
class FastSpeech2Loss(nn.Module):
    """ FastSpeech2 Loss """

    def __init__(self):
        super(FastSpeech2Loss, self).__init__()
        self.mse_loss = nn.MSELoss()
        self.mae_loss = nn.L1Loss()
        self.softmax = nn.Softmax(dim=1)

    def add_weight_loss(self, mel, mel_postnet):
        mel_norm = torch.norm(mel, p=float('inf'), dim=2)
        mel_postnet_norm = torch.norm(mel_postnet, p=float('inf'), dim=2)
        mel_scale = mel_norm#self.softmax(mel_norm)
        mel_postnet_scale = mel_postnet_norm#self.softmax(mel_postnet_norm)
        mel_scale = mel_scale.unsqueeze(2).repeat(1, 1, 80)
        #print(mel_postnet_scale.size())
        mel_postnet_scale = mel_postnet_scale.unsqueeze(2).repeat(1, 1, 80)
        #print(mel_postnet_scale.size())
        return mel_scale, mel_postnet_scale


    def forward(self, mel, mel_postnet, mel_target, src_mask, mel_mask, max_mel_len):
        #forward(self, log_d_predicted, log_d_target, p_predicted, p_target, e_predicted, e_target, mel, mel_postnet, mel_target, src_mask, mel_mask):
        '''
        log_d_target.requires_grad = False
        p_target.requires_grad = False
        e_target.requires_grad = False
        '''
        mel_target.requires_grad = False
        B, T, bi = mel_postnet.size()
        B_t, T_t, bi_t = mel_target.size()
        #print(mel_postnet.size(), mel_target.size())
        ### true mel legth > synthesized mel legth,  max_mel_len equal synthesized mel maxlegth
        if T_t>T:
            mel_target = mel_target[:, :T, :]
            mel_mask = mel_mask[:, :T]
        else:
            mel = mel[:,:T_t,:]
            mel_postnet = mel_postnet[:,:T_t,:]
            mel_mask = mel_mask[:, :T_t]

        '''
        log_d_predicted = log_d_predicted.masked_select(src_mask)
        log_d_target = log_d_target.masked_select(src_mask)
        p_predicted = p_predicted.masked_select(mel_mask)
        p_target = p_target.masked_select(mel_mask)
        e_predicted = e_predicted.masked_select(mel_mask)
        e_target = e_target.masked_select(mel_mask)
        '''

        #mel_scale, mel_postnet_scale = self.add_weight_loss(mel, mel_postnet)

        mel = mel.masked_select(mel_mask.unsqueeze(-1))
        mel_postnet = mel_postnet.masked_select(mel_mask.unsqueeze(-1))
        mel_target = mel_target.masked_select(mel_mask.unsqueeze(-1))

        #mel_scale = mel_scale.masked_select(mel_mask.unsqueeze(-1))
        #mel_postnet_scale = mel_postnet_scale.masked_select(mel_mask.unsqueeze(-1))
        

        
        t_mean = torch.mean(mel_target, keepdim=True, dim=-1)
        t_std = torch.std(mel_target, keepdim=True, dim=-1)

        s_mean = torch.mean(mel_postnet, keepdim=True, dim=-1)
        s_std = torch.std(mel_postnet, keepdim=True, dim=-1)        

        std_loss = self.mae_loss(s_std, t_std)+self.mae_loss(s_mean, t_mean)
        

        mel_loss = self.mse_loss(mel, mel_target)
        mel_postnet_loss = self.mse_loss(mel_postnet, mel_target)

#         class_loss = ReversalClassifier.loss(source_length, speaker, speaker_prediction) 

        '''
        d_loss = self.mae_loss(log_d_predicted, log_d_target)
        p_loss = self.mae_loss(p_predicted, p_target)
        e_loss = self.mae_loss(e_predicted, e_target)
        '''

        return mel_loss, mel_postnet_loss, std_loss#, class_loss#d_loss, p_loss, e_loss
