import torch
import torch.nn as nn
import torch.nn.functional as F

from transformer.Models import Encoder, Decoder
from transformer.Layers import PostNet, PreNet
from modules import VarianceAdaptor
from utils import get_mask_from_lengths
import hparams as hp
from classifier import ReversalClassifier

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class FastSpeech2(nn.Module):
    """ FastSpeech2 """

    def __init__(self, use_postnet=True, input_size=256):
        super(FastSpeech2, self).__init__()
        
        self.input_size = input_size
        self.prenet = PreNet(input_size, hp.prenet_size, hp.prenet_size, hp.encoder_dropout)
        self.embdnet = PreNet(hp.embd_dim, hp.embd_dim, hp.embd_dim, hp.embd_dropout)

        self.encoder1 = Encoder()
        self.encoder2 = Encoder()
        #self.variance_adaptor = VarianceAdaptor()

        self.decoder1 = Decoder()
        self.decoder2 = Decoder()
        self.mel_linear = nn.Linear(hp.decoder_hidden, hp.n_mel_channels)
        self.projection1 = nn.Linear(hp.decoder_hidden*2, hp.decoder_hidden)
        self.projection2 = nn.Linear(hp.decoder_hidden*2, hp.decoder_hidden)

#         self.reversal_classifier = ReversalClassifier(256, 256, hp.num_speaker)#现在的训练集有95人，测试集有9人

        self.use_postnet = use_postnet
#         if self.use_postnet:
        self.postnet = PostNet()

    def forward(self, src_seq, src_len, mel_len=None, max_src_len=None, max_mel_len=None, embeds=None, embeds_tar=None):  
#         with torch.no_grad():
        ### prenet 
#         input_feature = torch.cat((bnf, src_seq), -1)
        bnf_embedding = self.prenet(src_seq)

        # mask
        src_mask = get_mask_from_lengths(src_len, max_src_len)
        mel_mask = get_mask_from_lengths(mel_len, max_mel_len) #if mel_len is not None else None
        #print('src_mask', src_mask.shape)
        #print('src_seq', src_seq.shape)

        if hp.spk_embedding:
#             embeds = F.normalize(embeds, p=2, dim=-1)
            embeds = hp.embed_scale*embeds
            seq_length = bnf_embedding.size(1)
            #cat_embed = embed.repeat(1, seq_length, 1)#embed.unsqueeze(1).repeat(1, seq_length, 1)
            cat_embed = embeds.unsqueeze(1).repeat(1, seq_length, 1)
            cat_embed = self.embdnet(cat_embed)
            
            embeds_tar = hp.embed_scale*embeds_tar
            seq_length = bnf_embedding.size(1)
            #cat_embed = embed.repeat(1, seq_length, 1)#embed.unsqueeze(1).repeat(1, seq_length, 1)
            cat_embed_tar = embeds_tar.unsqueeze(1).repeat(1, seq_length, 1)
            cat_embed_tar = self.embdnet(cat_embed_tar)
            
#         encoder_input1 = bnf_embedding - cat_embed
        encoder_input1 = torch.cat((bnf_embedding, cat_embed), -1)
        encoder_input1 = self.projection1(encoder_input1)
        ## encoder
        encoder_output1 = self.encoder1(encoder_input1, src_mask)


        ### decoder 去说话人特征的output加同一个spk的embedding
#         decoder_input = encoder_output + cat_embed
#         decoder_input = torch.cat((encoder_output, cat_embed), -1)
#         decoder_input = self.projection_dec(decoder_input)
        decoder_input1 = encoder_output1
        decoder_output1 = self.decoder1(decoder_input1, src_mask)
        mel_output1 = self.mel_linear(decoder_output1)


        #de personalized output get encoded with embedding to get target mel
        encoder_input2 = torch.cat((decoder_output1, cat_embed_tar), -1)
        encoder_input2 = self.projection2(encoder_input2)
#         encoder_input2 = decoder_output1 + cat_embed_tar
        encoder_output2 = self.encoder2(encoder_input2, src_mask)
        decoder_input2 = encoder_output2
        decoder_output2 = self.decoder2(decoder_input2, src_mask)
        mel_output2 = self.mel_linear(decoder_output2)

        ### postnet
        if self.use_postnet:
            mel_output2_postnet = self.postnet(mel_output2) + mel_output2
        else:
            mel_output2_postnet = mel_output2

        return decoder_output1, mel_output1, mel_output2, mel_output2_postnet, src_mask, mel_mask

if __name__ == "__main__":
    # Test
    model = FastSpeech2(use_postnet=False)
    print(model)
    print(sum(param.numel() for param in model.parameters()))
