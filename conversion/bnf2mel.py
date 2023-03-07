import torch
import torch.nn as nn
import torch.nn.functional as F

from transformer.Models import Encoder, Decoder
from transformer.Layers import PostNet, PreNet
from modules import VarianceAdaptor
from utils import get_mask_from_lengths
import hp_bnf2mel as hp
from classifier import ReversalClassifier

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class bnf2mel(nn.Module):
    """ FastSpeech2 """

    def __init__(self, use_postnet=True):
        super(bnf2mel, self).__init__()
        
        self.prenet = PreNet(hp.bnf_size, hp.prenet_size, hp.prenet_size, hp.encoder_dropout)
        self.encoder = Encoder()
        #self.variance_adaptor = VarianceAdaptor()

        self.decoder = Decoder()
        self.mel_linear = nn.Linear(hp.decoder_hidden, hp.n_mel_channels)
        self.projection = nn.Linear(hp.decoder_hidden+hp.gvector_dim, hp.decoder_hidden)

        self.reversal_classifier = ReversalClassifier(256, 256, 174)#174

        self.use_postnet = use_postnet
        if self.use_postnet:
            self.postnet = PostNet()

    def forward(self, src_seq, src_len, mel_len=None, max_src_len=None, max_mel_len=None, embeds=None):
        
#         with torch.no_grad():
            ### prenet 
        bnf_embedding = self.prenet(src_seq)

        # mask
        src_mask = get_mask_from_lengths(src_len, max_src_len)
        mel_mask = get_mask_from_lengths(
            src_len, max_src_len) if mel_len is not None else None
        #print('src_mask', src_mask.shape)
        #print('src_seq', src_seq.shape)
        ## encoder
        encoder_output = self.encoder(bnf_embedding, src_mask)
#         speaker_prediction = self.reversal_classifier(encoder_output)

        if hp.spk_embedding:
            embed = F.normalize(embeds, p=2, dim=-1)
            seq_length = encoder_output.size(1)
            #cat_embed = embed.repeat(1, seq_length, 1)#embed.unsqueeze(1).repeat(1, seq_length, 1)
            cat_embed = embed.unsqueeze(1).repeat(1, seq_length, 1)
            encoder_output = torch.cat((encoder_output, cat_embed), -1)
            encoder_output = self.projection(encoder_output)


        ### decoder
        decoder_output = self.decoder(encoder_output, src_mask)
        mel_output = self.mel_linear(decoder_output)


        ### postnet
        if self.use_postnet:
            mel_output_postnet = self.postnet(mel_output) + mel_output
        else:
            mel_output_postnet = mel_output

        return mel_output, mel_output_postnet, src_mask, mel_mask


if __name__ == "__main__":
    # Test
    model = bnf2mel(use_postnet=False)
    print(model)
    print(sum(param.numel() for param in model.parameters()))
