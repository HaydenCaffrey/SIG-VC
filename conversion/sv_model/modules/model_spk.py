import torch, torch.nn as nn, numpy as np

from .front_resnet import ResNet34,BasicBlock as BasicBlock_resnet34,Res2Net,Bottle2neck,ResNet34SE,ResNet34_tendom,ResNeXt,ResNet34_dynamic
from .front_res2net import Res2Net34SE,Res2Net34SE_Rep
from .front_resnetfca import ResNet34FCA
from .front_transformer import Transformer_ASV
from .front_seres2net import SERes2Net34
from .front_res2net_v1b import res2net50_v1b
from .front_cbam import resnet34_cbam
from .front_tdnn import FTDNN,ETDNN,TDNNLayer
from .front_complex_resnet import cResNet34,convResNet34
from .pooling import StatsPool, ScaleDotProductAttention,TemporalStatsPool

class cResNet34StatsPool(nn.Module):

    def __init__(self, in_planes, embedding_size, dropout=0.5, **kwargs):

        super(cResNet34StatsPool, self).__init__()
        self.front = cResNet34(in_planes, **kwargs)
        self.pool = StatsPool()
        self.bottleneck = nn.Linear(in_planes*8*2, embedding_size)
        self.drop = nn.Dropout(dropout) if dropout else None
        
    def forward(self, x):
        x = self.front(x)
        x = self.pool(x)
        x = self.bottleneck(x)
        if self.drop:
            x = self.drop(x)
        return x
    
class ConvResNet34StatsPool(nn.Module):

    def __init__(self, in_planes, embedding_size, dropout=0.5, **kwargs):

        super(ConvResNet34StatsPool, self).__init__()
        self.front = convResNet34(in_planes, **kwargs)
        self.pool = StatsPool()
        self.bottleneck = nn.Linear(in_planes*8*2, embedding_size)
        self.drop = nn.Dropout(dropout) if dropout else None
        
    def forward(self, x):
        x = self.front(x)
        x = self.pool(x)
        x = self.bottleneck(x)
        if self.drop:
            x = self.drop(x)
        return x


class Transformer_SPEECH(nn.Module):

    def __init__(self, embedding_size, length, dropout=0, **kwargs):

        super(Transformer_SPEECH, self).__init__()
        self.front = Transformer_ASV(N=3,d_model=512, d_ff=128, h=4, dropout=dropout)
        self.pool = StatsPool()
        self.bottleneck = nn.Linear(512*2, embedding_size)
#         self.bn = nn.BatchNorm2d(embedding_size)
        self.drop = nn.Dropout(0) if dropout else None
        
    def forward(self, x):
        x = self.front(x)
        x = self.pool(x)
        x = self.bottleneck(x)
#         x = self.bn(x)
        if self.drop:
            x = self.drop(x)
        return x

class ResNeXtStatsPool(nn.Module):

    def __init__(self, in_planes, embedding_size, dropout=0.5, **kwargs):

        super(ResNeXtStatsPool, self).__init__()
        self.front = ResNeXt(in_planes, **kwargs)
        self.pool = StatsPool()
        self.bottleneck = nn.Linear(in_planes*2, embedding_size)
        self.drop = nn.Dropout(dropout) if dropout else None
        
    def forward(self, x):
        x = self.front(x.unsqueeze(dim=1))
        x = self.pool(x)
        x = self.bottleneck(x)
        if self.drop:
            x = self.drop(x)
        return x
    
class DynamicResNetStatsPool(nn.Module):

    def __init__(self, in_planes, embedding_size, dropout=0.5, **kwargs):

        super(DynamicResNetStatsPool, self).__init__()
        self.front = ResNet34_dynamic(in_planes, **kwargs)
        self.pool = StatsPool()
        self.bottleneck = nn.Linear(in_planes*8*2, embedding_size)
        self.drop = nn.Dropout(dropout) if dropout else None
        
    def forward(self, x):
        x = self.front(x.unsqueeze(dim=1))
        x = self.pool(x)
        x = self.bottleneck(x)
        
        if self.drop:
            x = self.drop(x)
        return x

def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)
    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)


def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias:
            nn.init.constant_(m.bias, 0.0)


class Standard_TDNN_GE2E(nn.Module):
    def __init__(self, in_planes, embedding_size, dropout=0.0,hidden_features=512, **kwargs):

        super(Standard_TDNN_GE2E, self).__init__()
        self.front1 = TDNNLayer(input_dim=in_planes,output_dim=hidden_features,context_size=5, stride=1, dilation=1, padding=0)
        self.front2 = TDNNLayer(input_dim=hidden_features,output_dim=hidden_features,context_size=3, stride=2, dilation=1, padding=0)
        self.front3 = TDNNLayer(input_dim=hidden_features,output_dim=hidden_features,context_size=3, stride=3, dilation=1, padding=0)
        self.front4 = TDNNLayer(input_dim=hidden_features,output_dim=hidden_features,context_size=1, stride=1, dilation=1, padding=0)
        self.front5 = TDNNLayer(input_dim=hidden_features,output_dim=1500,context_size=1, stride=1, dilation=1, padding=0)
        self.front = nn.Sequential(self.front1, self.front2 ,self.front3 ,self.front4 ,self.front5 )
        self.front6 = nn.Linear(3000, embedding_size,bias=False)
        self.front7 = nn.Linear(embedding_size, 256,bias=False)
        self.drop = nn.Dropout(dropout) if dropout else None
        self.pool = TDNN_StatsPool()
        
    def forward(self, x):
        x = self.front(x.transpose(1,2))
        x = self.pool(x.transpose(1,2))
        x = self.front6(x)
        x = self.front7(x)
        return x
    
def nonLinearAct():
    return nn.LeakyReLU()

class TDNN_StatsPool(nn.Module):

    def __init__(self, floor=1e-10, bessel=False):
        super(TDNN_StatsPool, self).__init__()
        self.floor = floor
        self.bessel = bessel

    def forward(self, x):
        means = torch.mean(x, dim=1)
        _, t, _ = x.shape
        if self.bessel:
            t = t - 1
        residuals = x - means.unsqueeze(1)
        numerator = torch.sum(residuals**2, dim=1)
        stds = torch.sqrt(torch.clamp(numerator, min=self.floor)/t)
        x = torch.cat([means, stds], dim=1)
        return x

class BIG_ETDNNStatsPool(nn.Module):

    def __init__(self, in_planes, embedding_size, dropout=0.0,hidden_features=1024, **kwargs):

        super(BIG_ETDNNStatsPool, self).__init__()
        self.front = ETDNN(features_per_frame=in_planes,
                    hidden_features=hidden_features,
                    dropout_p=dropout,
                    batch_norm=True)
        self.pool = TDNN_StatsPool()
        self.bottleneck = nn.Linear(hidden_features*6, embedding_size)
        self.drop = nn.Dropout(dropout) if dropout else None
    
    def forward(self, x):
        x = self.front(x)
        x = self.pool(x)
        x = self.bottleneck(x)
        if self.drop:
            x = self.drop(x)
        return x

class ResNet34StatsPool(nn.Module):

    def __init__(self, in_planes, embedding_size, dropout=0.5, **kwargs):

        super(ResNet34StatsPool, self).__init__()
        self.front = ResNet34(in_planes, **kwargs)
        self.pool = StatsPool()
        self.bottleneck = nn.Linear(in_planes*8*2, embedding_size)
        self.drop = nn.Dropout(dropout) if dropout else None
        
    def forward(self, x):
        x = self.front(x.unsqueeze(dim=1))
        x = self.pool(x)
        x = self.bottleneck(x)
        if self.drop:
            x = self.drop(x)
        return x
    
class ResNet34TandomStatsPool(nn.Module):

    def __init__(self, in_planes, embedding_size, dropout=0.5, **kwargs):

        super(ResNet34TandomStatsPool, self).__init__()
        self.front = ResNet34_tendom(in_planes, **kwargs)
        self.pool = StatsPool()
        self.bottleneck = nn.Linear((in_planes*8+in_planes*4+in_planes*2+in_planes)*2, embedding_size)
        self.drop = nn.Dropout(dropout) if dropout else None
        
    def forward(self, x):
        x1,x2,x3,x4 = self.front(x.unsqueeze(dim=1))
        x1 = self.pool(x1)
        x2 = self.pool(x2)
        x3 = self.pool(x3)
        x4 = self.pool(x4)
        x = torch.cat([x1,x2,x3,x4], dim=1)
        x = self.bottleneck(x)
        if self.drop:
            x = self.drop(x)
        return x
    
class ResNet34CBAMStatsPool(nn.Module):

    def __init__(self, in_planes, embedding_size, dropout=0.5, **kwargs):
        
        super(ResNet34CBAMStatsPool, self).__init__()
        self.front = resnet34_cbam(in_planes=in_planes)
        self.pool = StatsPool()
        self.bottleneck = nn.Linear(in_planes*8*2, embedding_size)
        self.drop = nn.Dropout(dropout) if dropout else None
        
    def forward(self, x):
        x = self.front(x.unsqueeze(dim=1))
        x = self.pool(x)
        x = self.bottleneck(x)
        if self.drop:
            x = self.drop(x)
        return x
    
class ResNet34FCAStatsPool(nn.Module):

    def __init__(self, in_planes, embedding_size, dropout=0.5, **kwargs):

        super(ResNet34FCAStatsPool, self).__init__()
        self.front = ResNet34FCA(in_planes, **kwargs)
        self.pool = StatsPool()
        self.bottleneck = nn.Linear(in_planes*8*2, embedding_size)
        self.drop = nn.Dropout(dropout) if dropout else None
        
    def forward(self, x):
        x = self.front(x.unsqueeze(dim=1))
        x = self.pool(x)
        x = self.bottleneck(x)
        if self.drop:
            x = self.drop(x)
        return x
    
class Res2Net34SEStatsPool(nn.Module):

    def __init__(self, in_planes, embedding_size, dropout=0.5, **kwargs):
        
        super(Res2Net34SEStatsPool, self).__init__()
        self.front = Res2Net34SE(in_planes=in_planes, baseWidth=26, scale=8, **kwargs) #scale = 4,8 可以被in_planes 整除
        self.pool = StatsPool()
        self.bottleneck = nn.Linear(in_planes*8*2, embedding_size)
        self.drop = nn.Dropout(dropout) if dropout else None
        
    def forward(self, x):
        x = self.front(x.unsqueeze(dim=1))
        x = self.pool(x)
        x = self.bottleneck(x)
        if self.drop:
            x = self.drop(x)
        return x
    
    
class Res2Net34SERepStatsPool(nn.Module):

    def __init__(self, in_planes, embedding_size, dropout=0.5, **kwargs):
        
        super(Res2Net34SERepStatsPool, self).__init__()
        self.front = Res2Net34SE_Rep(in_planes=in_planes, baseWidth=26, scale=8, **kwargs) #scale = 4,8 可以被in_planes 整除
        self.pool = StatsPool()
        self.bottleneck = nn.Linear(in_planes*8*2, embedding_size)
        self.drop = nn.Dropout(dropout) if dropout else None
        
    def forward(self, x):
        x = self.front(x.unsqueeze(dim=1))
        x = self.pool(x)
        x = self.bottleneck(x)
        if self.drop:
            x = self.drop(x)
        return x
    
    
class Res2Net50_v1b_StatsPool(nn.Module):

    def __init__(self, in_planes, embedding_size, dropout=0, **kwargs):
        
        super(Res2Net50_v1b_StatsPool, self).__init__()
        self.front = res2net50_v1b() #in_planes=64, baseWidth=26, scale=4
        self.pool = StatsPool()
        self.bottleneck = nn.Linear(in_planes*8*2*4, embedding_size)
        self.drop = nn.Dropout(dropout) if dropout else None
        
    def forward(self, x):
        x = self.front(x.unsqueeze(dim=1))
        x = self.pool(x)
        x = self.bottleneck(x)
        if self.drop:
            x = self.drop(x)
        return x
    
class SERes2Net34StatsPool(nn.Module):

    def __init__(self, in_planes, embedding_size, dropout=0.5, **kwargs):
        
        super(SERes2Net34StatsPool, self).__init__()
        self.front = SERes2Net34(input_channel=1,in_planes=in_planes) #scale = 4,8 可以被in_planes 整除
        self.pool = StatsPool()
        self.bottleneck = nn.Linear(in_planes*8*2, embedding_size)
        self.drop = nn.Dropout(dropout) if dropout else None
        
    def forward(self, x):
        x = self.front(x.unsqueeze(dim=1))
        x = self.pool(x)
        x = self.bottleneck(x)
        if self.drop:
            x = self.drop(x)
        return x
    
class ResNet34StatsPool_BNneck(nn.Module):

    def __init__(self, in_planes, embedding_size, dropout=0.5, **kwargs):

        super(ResNet34StatsPool_BNneck, self).__init__()
        self.front = ResNet34(in_planes, **kwargs)
        self.pool = StatsPool()
        self.bottleneck = nn.Linear(in_planes*8*2, embedding_size)
        self.bnnneck = nn.BatchNorm1d(embedding_size)
        self.bnnneck.bias.requires_grad_(False)
        
        self.drop = nn.Dropout(dropout) if dropout else None
        
        self.bnnneck.apply(weights_init_kaiming)

    def forward(self, x):
        x = self.front(x.unsqueeze(dim=1))
        x = self.pool(x)
        x_tri = self.bnneck(x)
        embd = self.bottleneck(x)
        
        if self.drop:
            embd = self.drop(embd)
        return x_tri,embd

class ResNet34SEStatsPool(nn.Module):
    def __init__(self, in_planes, embedding_size, dropout=0.5, **kwargs):
        super(ResNet34SEStatsPool, self).__init__()
        self.front = ResNet34SE(in_planes, **kwargs)
        self.pool = StatsPool()
        self.bottleneck = nn.Linear(in_planes*8*2, embedding_size)
        self.drop = nn.Dropout(dropout) if dropout else None
        
    def forward(self, x):
#         with torch.no_grad():
        x = x.transpose(1,2)
        x = self.front(x.unsqueeze(dim=1))
        x = self.pool(x)
        x = self.bottleneck(x)
        if self.drop:
            x = self.drop(x)
        return x    
    
class Res2NetStatsPool(nn.Module):

    def __init__(self, in_planes, embedding_size, dropout=0.5, se=False,**kwargs):

        super(Res2NetStatsPool, self).__init__()
        self.front = Res2Net(Bottle2neck, [3, 3, 3], inplanes=in_planes,baseWidth = 13, scale = 4, se=se, **kwargs)
        self.pool = StatsPool()
        self.bottleneck = nn.Linear(in_planes*4*2, embedding_size)
        self.drop = nn.Dropout(dropout) if dropout else None
        
    def forward(self, x):
        x = self.front(x.unsqueeze(dim=1))
        x = self.pool(x)
        x = self.bottleneck(x)
        if self.drop:
            x = self.drop(x)
        return x
    

class FTDNNStatsPool(nn.Module):

    def __init__(self, in_planes, embedding_size, total_step=200, dropout=0.5, factorize_step_size=4):

        super(FTDNNStatsPool, self).__init__()
        self.front = FTDNN(in_planes)
        self.pool = StatsPool()
        self.bottleneck = nn.Linear(4096, embedding_size)
        self.bn = nn.BatchNorm1d(embedding_size)
        self.nl = nn.LeakyReLU()
        self.drop = nn.Dropout(dropout) if dropout else None
        
        self.step = 0
        self.drop_schedule = np.interp(np.linspace(0, 1, total_step), [0, 0.5, 1], [0, 0.5, 0])
        self.factorize_step_size = factorize_step_size
        
    def forward(self, x):
        if self.training:
            self.front.set_dropout_alpha(self.drop_schedule[self.step])
            if self.step % self.factorize_step_size == 1:
                self.front.step_ftdnn_layers()
            self.step += self.step
        
        x = self.front(x)
        x = self.pool(x)
        x = self.bottleneck(x)
        x = self.nl(x)
        x = self.bn(x)
        if self.drop:
            x = self.drop(x)
        return x

    
class ResNet34SDPAttPool(nn.Module):

    def __init__(self, in_planes, embedding_size, dropout=0.5, **kwargs):

        super(ResNet34SDPAttPool, self).__init__()
        self.front = ResNet34(in_planes, **kwargs)
        self.pool = ScaleDotProductAttention(in_planes*8)
        self.bottleneck = nn.Linear(in_planes*8, embedding_size)
        self.drop = nn.Dropout(dropout) if dropout else None

    def forward(self, x):
        x = self.front(x.unsqueeze(dim=1))    ### batch x channel x freq x time
        x = x.mean(dim=2).transpose(1, 2)    ### batch x time x feat_dim
        x = self.pool(x)
        x = self.bottleneck(x)
        if self.drop:
            x = self.drop(x)
        return x
    
####################################################################
##### ECAPA_TDNN #######################################
####################################################################

import torch
from torch import nn
from torch.nn import Parameter
import torch.nn.functional as F

class SE_Res2Block(nn.Module):
    
    def __init__(self,k=3,d=2,s=8,channel=512,bottleneck=128):
        super(SE_Res2Block,self).__init__()
        self.k = k
        self.d = d
        self.s = s
        if self.s == 1:
            self.nums = 1
        else:
            self.nums = self.s - 1
            
        self.channel = channel
        self.bottleneck = bottleneck
        
        self.conv1 = nn.Conv1d(self.channel,self.channel,kernel_size=1,dilation=1)
        self.bn1 = nn.BatchNorm1d(self.channel)
        
        self.convs = []
        self.bns = []
        for i in range(self.s):
            self.convs.append(nn.Conv1d(int(self.channel/self.s), int(self.channel/self.s), kernel_size=self.k, dilation=self.d, padding=self.d, bias=False,padding_mode='reflect'))
            self.bns.append(nn.BatchNorm1d(int(self.channel/self.s)))
            
        self.convs = nn.ModuleList(self.convs)
        self.bns = nn.ModuleList(self.bns)
        
        self.conv3 = nn.Conv1d(self.channel,self.channel,kernel_size=1,dilation=1)
        self.bn3 = nn.BatchNorm1d(self.channel)
        
        self.fc1 = nn.Linear(self.channel,self.bottleneck,bias=True)
        self.fc2 = nn.Linear(self.bottleneck,self.channel,bias=True)
        
    def forward(self,x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))

        spx = torch.split(out, int(self.channel/self.s), 1)
        for i in range(1,self.nums+1):
            if i==1:
                sp = spx[i]
            else:
                sp = sp + spx[i]
            sp = self.convs[i](sp)
            sp = F.relu(self.bns[i](sp))
            if i==1:
                out = sp
            else:
                out = torch.cat((out, sp), 1)

        if self.s != 1 :
            out = torch.cat((out, spx[0]),1)
        
        out = F.relu(self.bn3(self.conv3(out)))
        out_mean = torch.mean(out,dim=2)
        s_v = torch.sigmoid(self.fc2(F.relu(self.fc1(out_mean))))
        out = out * s_v.unsqueeze(-1)
        out += residual
        #out = F.relu(out)
        return out


class Classic_Attention(nn.Module):
    def __init__(self,input_dim, embed_dim, attn_dropout=0.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.attn_dropout = attn_dropout
        self.lin_proj = nn.Linear(input_dim,embed_dim)
        self.v = torch.nn.Parameter(torch.randn(embed_dim))
    
    def forward(self,inputs):
        lin_out = self.lin_proj(inputs)
        v_view = self.v.unsqueeze(0).expand(lin_out.size(0), len(self.v)).unsqueeze(2)
        attention_weights = torch.tanh(lin_out.bmm(v_view).squeeze(-1))
        attention_weights_normalized = F.softmax(attention_weights,1)
        #attention_weights_normalized = F.softmax(attention_weights)
        return attention_weights_normalized

class Attentive_Statictics_Pooling(nn.Module):
    
    def __init__(self,channel=1536,R_dim_self_att=128):
        super(Attentive_Statictics_Pooling,self).__init__()
        
        self.attention = Classic_Attention(channel,R_dim_self_att)
    
    def weighted_sd(self,inputs,attention_weights, mean):
        el_mat_prod = torch.mul(inputs,attention_weights.unsqueeze(2).expand(-1,-1,inputs.shape[-1]))
        hadmard_prod = torch.mul(inputs,el_mat_prod)
        variance = torch.sum(hadmard_prod,1) - torch.mul(mean,mean)
        return variance    
    
    def stat_attn_pool(self,inputs,attention_weights):
        el_mat_prod = torch.mul(inputs,attention_weights.unsqueeze(2).expand(-1,-1,inputs.shape[-1]))
        mean = torch.mean(el_mat_prod,1)
        variance = self.weighted_sd(inputs,attention_weights,mean)
        stat_pooling = torch.cat((mean,variance),1)
        return stat_pooling
    
    def forward(self,x):
        attn_weights = self.attention(x)
        stat_pool_out = self.stat_attn_pool(x,attn_weights)
        
        return stat_pool_out
    
class ECAPA_TDNN(nn.Module):
    
    def __init__(self,in_planes=80,hidden_dim=512,scale=8,bottleneck=128,embedding_size=192):
        
        super(ECAPA_TDNN,self).__init__()
        
        self.in_dim = in_planes
        self.hidden_dim = hidden_dim
        self.scale = scale
        self.bottleneck = bottleneck
        self.embedding_size = embedding_size
        
        self.conv1 = nn.Conv1d(in_planes,hidden_dim,kernel_size=5,dilation=1)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.block1 = SE_Res2Block(k=3,d=2,s=self.scale,channel=self.hidden_dim,bottleneck=self.bottleneck)
        self.block2 = SE_Res2Block(k=3,d=3,s=self.scale,channel=self.hidden_dim,bottleneck=self.bottleneck)
        self.block3 = SE_Res2Block(k=3,d=4,s=self.scale,channel=self.hidden_dim,bottleneck=self.bottleneck)
        self.conv2 = nn.Conv1d(self.hidden_dim*3,self.hidden_dim*3,kernel_size=1,dilation=1)
        
        self.ASP = Attentive_Statictics_Pooling(channel=self.hidden_dim*3,R_dim_self_att=self.bottleneck)
        self.bn2 = nn.BatchNorm1d(self.hidden_dim*3*2)
        
        self.fc = nn.Linear(self.hidden_dim*3*2,self.embedding_size)
        self.bn3 = nn.BatchNorm1d(self.embedding_size)
        
    def forward(self,x):
        x = x.transpose(1,2)
        y = F.relu(self.bn1(self.conv1(x)))
        y_1 = self.block1(y)
        y_2 = self.block2(y_1)
        y_3 = self.block3(y_2)
        out = torch.cat((y_1, y_2,y_3), 1)
        out = F.relu(self.conv2(out))
        out = self.bn2(self.ASP(out.transpose(1,2)))
        out = self.bn3(self.fc(out))
        return out

####################################################################
##### ResNet Block attention #######################################
####################################################################

class ResNet34AttAvgstdDropNet(nn.Module):

    def __init__(self,front_type='bam', attention='channel',input_channel=1, in_planes=64, embedding_size=64,dropout_rate=0):
       
        super(ResNet34AttAvgstdDropNet, self).__init__()
        if front_type=='se' and attention=='channel':
            self.front = se_resnet34(norm='bn', attention='channel') 
        elif front_type=='bam' and attention=='channel':
            self.front = bam_resnet34(norm='bn', attention='channel')
        elif front_type=='bam' and attention=='spatial':
            self.front = bam_resnet34(norm='bn', attention='spatial')
        elif front_type=='bam' and attention=='joint':
            self.front = bam_resnet34(norm='bn', attention='joint')
        elif front_type=='cbam' and attention=='channel':
            self.front = cbam_resnet34(norm='bn', attention='channel')
        elif front_type=='cbam' and attention=='spatial':
            self.front = cbam_resnet34(norm='bn', attention='spatial')
        elif front_type=='cbam' and attention=='joint':
            self.front = cbam_resnet34(norm='bn', attention='joint')
        else:
            raise Exception('Unknown model')
        
        print('model:ResNet_%s and attention: %s' %(front_type,attention))
        self.pool = StatsPool()
        self.bottleneck = nn.Linear(in_planes*8*2, embedding_size)
        self.drop = nn.Dropout(dropout_rate) if dropout_rate else None

    def forward(self, x):
        out = self.front(x.unsqueeze(dim=1))
        out = self.pool(out)
        out = self.bottleneck(out)
        if self.drop:
            out = self.drop(out)
        return out

import torch.nn as nn
import math
import torch
import torch.nn.functional as F
import pdb

# Utility function

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=True)

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class Norm(nn.Module):
    def __init__(self, name, n_feats):
        super(Norm, self).__init__()
        assert name in ['bn', 'gn', 'gbn', 'none']
        if name == 'bn':
            self.norm = nn.BatchNorm2d(n_feats)
        elif name == 'gn':
            self.norm = nn.GroupNorm(32, n_feats)
        elif name == 'gbn':
            self.norm = nn.Sequential(nn.GroupNorm(32, n_feats, affine=False),nn.BatchNorm2d(n_feats))
        elif name == 'none':
            pass
        self.name = name

    def forward(self, x):
        if self.name == 'none':
            return x
        else:
            return self.norm(x)

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class BamSpatialAttention(nn.Module):
    def __init__(self,channel,reduction = 16, dilation_ratio =2):
        super(BamSpatialAttention,self).__init__()

        self.body = nn.Sequential(
            nn.Conv2d(channel,channel//reduction,1),

            nn.BatchNorm2d(channel//reduction),
            nn.Conv2d(channel//reduction,channel//reduction,3,padding=dilation_ratio,dilation=dilation_ratio),
            nn.BatchNorm2d(channel // reduction),
            nn.ReLU(True),

            nn.BatchNorm2d(channel // reduction),
            nn.Conv2d(channel // reduction, channel // reduction, 3, padding=dilation_ratio, dilation=dilation_ratio),
            nn.BatchNorm2d(channel // reduction),
            nn.ReLU(True),

            nn.Conv2d(channel//reduction,1,1)
        )
    def forward(self, x):
        return self.body(x).expand_as(x)


class BamChannelAttention(nn.Module):
    def __init__(self,channel,reduction = 16):
        super(BamChannelAttention,self).__init__()
        self.avgPool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channel,channel//reduction,1),
            #nn.BatchNorm2d(channel//reduction)
            nn.ReLU(inplace=True),
            nn.Conv2d(channel//reduction,channel,1),
        )
    def forward(self,x):
        out = self.avgPool(x)
        out = self.fc(out)
        return out.expand_as(x)




class CBamSpatialAttention(nn.Module):
    def __init__(self,channel,reduction = 16):
        super(CBamSpatialAttention,self).__init__()
        kernel_size = 5
        self.att = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size, stride=1, padding=(kernel_size-1)//2),
            nn.BatchNorm2d(1),
        )

    def forward(self, x):
        out = self._PoolAlongChannel(x)
        out = self.att(out)
        out = F.sigmoid(out)
        return x*out

    def _PoolAlongChannel(self,x):
        return torch.cat((torch.max(x,1)[0].unsqueeze(1), torch.mean(x,1).unsqueeze(1)), dim=1)


class CBamChannelAttention(nn.Module):
    def __init__(self,channel,reduction = 16):
        super(CBamChannelAttention,self).__init__()
        self.channel = channel
        self.fc = nn.Sequential(
            Flatten(),
            nn.Linear(self.channel,self.channel//reduction),
            nn.ReLU(),
            nn.Linear(self.channel//reduction,self.channel)
        )
    def forward(self, x):
        avgPool = F.avg_pool2d(x,(x.size(2),x.size(3)),stride =  (x.size(2),x.size(3)))
        out1 = self.fc(avgPool)
        maxPool = F.max_pool2d(x,(x.size(2),x.size(3)),stride =  (x.size(2),x.size(3)))
        out2 = self.fc(maxPool)
        out = out1 + out2
        att = F.sigmoid(out).unsqueeze(2).unsqueeze(3).expand_as(x)
        return x*att



# Attention module
class SE_Attention_Layer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SE_Attention_Layer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
                nn.Conv2d(channel, channel // reduction, 1),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // reduction, channel, 1)
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.fc(y)
        y = F.sigmoid(y)
        return  x*y.expand_as(x)

class BAM_Attention_Layer(nn.Module):
    def __init__(self, channel, att = 'both', reduction=16):
        super(BAM_Attention_Layer, self).__init__()
        self.att = att
        self.channelAtt =None
        self.spatialAtt =None
        if att == 'both' or att == 'c':
            self.channelAtt = BamChannelAttention(channel,reduction)
        if att == 'both' or att == 's':
            self.spatialAtt = BamSpatialAttention(channel,reduction)

    def forward(self, x):
        if self.att =='both':
            y1 = self.spatialAtt(x)
            y2 = self.channelAtt(x)
            y = y1+ y2
        elif self.att =='c':
            y = self.channelAtt(x)
        elif self.att =='s':
            y = self.spatialAtt(x)
        return (1 +F.sigmoid(y))*x

class CBAM_Attention_Layer(nn.Module):
    def __init__(self, channel,att = 'both', reduction=16):
        super(CBAM_Attention_Layer, self).__init__()
        self.att = att
        self.channelAtt = None
        self.spatialAtt = None
        if att == 'both' or att == 'c':
            self.channelAtt = CBamChannelAttention(channel,reduction)
        if att == 'both' or att == 's':
            self.spatialAtt = CBamSpatialAttention(channel,reduction)



    def forward(self, x):
        if self.att =='both':
            y = self.channelAtt(x)
            y = self.spatialAtt(y)
        elif self.att =='c':
            y = self.channelAtt(x)
        elif self.att =='s':
            y = self.spatialAtt(x)
        return y

# Blocks

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, attention='no', base_width= 64, t_norm = 'bn'):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=True)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=True)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=True)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.sigmoid = nn.Sigmoid()
        if attention == 'se':
            self.att = SE_Attention_Layer(planes * 4)
        elif attention == 'c_bam':
            self.att = None
        elif attention == 's_bam':
            self.att = None
        elif attention == 'j_bam':
            self.att = None
        elif attention == 'c_cbam':
            self.att = CBAM_Attention_Layer(planes * 4,'c')
        elif attention == 's_cbam':
            self.att = CBAM_Attention_Layer(planes * 4,'s')
        elif attention == 'j_cbam':
            self.att = CBAM_Attention_Layer(planes * 4,'both')
        elif attention == 'no':
            self.att = None
        else:
            raise Exception('Unknown att type')

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.att is not None:
            out = self.att(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, t_norm='bn', attention ='no'):
        super(BasicBlock, self).__init__()
        if t_norm == 'bn':
            norm_layer = nn.BatchNorm2d
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')

        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, width,stride=stride)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width)
        self.bn2 = norm_layer(width)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

        if attention == 'se':
            self.att = SE_Attention_Layer(planes)
        elif attention == 'c_bam':
            self.att = None
        elif attention == 's_bam':
            self.att = None
        elif attention == 'j_bam':
            self.att = None
        elif attention == 'c_cbam':
            self.att = CBAM_Attention_Layer(width,'c')
        elif attention == 's_cbam':
            self.att = CBAM_Attention_Layer(width,'s')
        elif attention == 'j_cbam':
            self.att = CBAM_Attention_Layer(width,'both')
        elif attention == 'no':
            self.att = None
        else:
            raise Exception('Unknown att type')


    def forward(self, x):

        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.att is not None:
            out = self.att(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        #pdb.set_trace()
        out += identity
        out = self.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, layers, norm='bn', attention='no'):
        super(ResNet, self).__init__()
        self.inplanes = 64
        self.size = 64
        self.norm = norm


        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=1, padding=3, bias=True)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, 64, layers[0], stride=1, attention=attention)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=1, attention=attention)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, attention=attention)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, attention=attention)


        if attention == 'c_bam':
            self.bam1 = BAM_Attention_Layer(64*block.expansion,'c')
            self.bam2 = BAM_Attention_Layer(128*block.expansion,'c')
            self.bam3 = BAM_Attention_Layer(256*block.expansion,'c')
        elif attention == 'j_bam':
            self.bam1 = BAM_Attention_Layer(64*block.expansion,'both')
            self.bam2 = BAM_Attention_Layer(128*block.expansion,'both')
            self.bam3 = BAM_Attention_Layer(256*block.expansion,'both')
        elif attention == 's_bam':
            self.bam1 = BAM_Attention_Layer(64*block.expansion,'s')
            self.bam2 = BAM_Attention_Layer(128*block.expansion,'s')
            self.bam3 = BAM_Attention_Layer(256*block.expansion,'s')
        else:
            self.bam1 = None
            self.bam2 = None
            self.bam3 = None



        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1, attention='no'):
        downsample = None
        if  stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=True),
                Norm(self.norm, planes * block.expansion),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, base_width=self.size, t_norm=self.norm, attention=attention))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, base_width=self.size, t_norm=self.norm, attention=attention))
        # append att layer to the stage
        #layers.append(block(self.inplanes, planes, attention=attention, size=size))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.layer1(x)
        if not self.bam1 is None:
            x = self.bam1(x)

        x = self.layer2(x)
        if not self.bam1 is None:
            x = self.bam2(x)

        x = self.layer3(x)
        if not self.bam1 is None:
            x = self.bam3(x)

        x = self.layer4(x)
        return x


def se_resnet50(attention="channel",norm = 'bn',**kwargs):
    if attention == "channel":
        model = ResNet(Bottleneck, [3, 4, 6, 3],norm, 'se', **kwargs)
    else:
        raise Exception("SEnet only support channel attention")
    return model


def bam_resnet50(attention="joint",norm = 'bn',**kwargs):
    if attention == "channel":
        model = ResNet(Bottleneck, [3, 4, 6, 3], norm, 'c_bam', **kwargs)
    elif attention == "spatial":
        model = ResNet(Bottleneck, [3, 4, 6, 3], norm, 's_bam', **kwargs)
    elif attention == "joint":
        model = ResNet(Bottleneck, [3, 4, 6, 3], norm, 'j_bam', **kwargs)
    else:
        raise Exception("Unknown attention for BAM")
    return model

def cbam_resnet50(attention="joint",norm = 'bn',**kwargs):
    if attention == "channel":
        model = ResNet(Bottleneck, [3, 4, 6, 3], norm, 'c_cbam', **kwargs)
    elif attention == "spatial":
        model = ResNet(Bottleneck, [3, 4, 6, 3], norm, 's_cbam', **kwargs)
    elif attention == "joint":
        model = ResNet(Bottleneck, [3, 4, 6, 3], norm, 'j_cbam', **kwargs)
    else:
        raise Exception("Unknown attention for CBAM")
    return model


def resnet50(attention="no",norm = 'bn',**kwargs):
    if attention == "no":
        model = ResNet(Bottleneck, [3, 4, 6, 3], norm,'no', **kwargs)
    else:
        raise Exception("Unknown attention for baseline resnet")
    return model


def se_resnet34(attention="channel",norm = 'bn',**kwargs):
    if attention == "channel":
        model = ResNet(BasicBlock, [3, 4, 6, 3],norm, 'se', **kwargs)
    else:
        raise Exception("SEnet only support channel attention")
    return model


def bam_resnet34(attention="joint",norm = 'bn',**kwargs):
    if attention == "channel":
        model = ResNet(BasicBlock, [3, 4, 6, 3], norm, 'c_bam', **kwargs)
    elif attention == "spatial":
        model = ResNet(BasicBlock, [3, 4, 6, 3], norm, 's_bam', **kwargs)
    elif attention == "joint":
        model = ResNet(BasicBlock, [3, 4, 6, 3], norm, 'j_bam', **kwargs)
    else:
        raise Exception("Unknown attention for BAM")
    return model

def cbam_resnet34(attention="joint",norm = 'bn',**kwargs):
    if attention == "channel":
        model = ResNet(BasicBlock, [3, 4, 6, 3], norm, 'c_cbam', **kwargs)
    elif attention == "spatial":
        model = ResNet(BasicBlock, [3, 4, 6, 3], norm, 's_cbam', **kwargs)
    elif attention == "joint":
        model = ResNet(BasicBlock, [3, 4, 6, 3], norm, 'j_cbam', **kwargs)
    else:
        raise Exception("Unknown attention for CBAM")
    return model

##############################
#### TRANSFORMER RES2NET ####
#############################

class SelfAttention(nn.Module):
    def __init__(self, k, heads = 8):
        super().__init__()
        self.k, self.heads = k, heads
        
        # input 维度为k（embedding结果），map成一个k*heads维度的矩阵
        self.tokeys    = nn.Linear(k, k * heads, bias = False)
        self.toqueries = nn.Linear(k, k * heads, bias = False)
        self.tovalues  = nn.Linear(k, k * heads, bias = False)
        # 在通过线性转换把维度压缩到 k
        self.unifyheads = nn.Linear(heads * k, k)
    
    def forward(self, x):
        b, t, k = x.size()
        h = self.heads

        queries = self.toqueries(x).view(b, t, h, k)
        keys    = self.tokeys(x).view(b, t, h, k)
        values  = self.tovalues(x).view(b, t, h, k)
        
        # 把 head 压缩进 batch的dimension
        queries = queries.transpose(1, 2).contiguous().view(b * h, t, k)
        keys    = keys.transpose(1, 2).contiguous().view(b * h, t, k)
        values  = values.transpose(1, 2).contiguous().view(b * h, t, k)
        
        # 这等效于对点积进行normalize
        queries = queries / (k ** (1/4))
        keys = keys / (k ** (1/4))
        # 矩阵相乘
        dot  = torch.bmm(queries, keys.transpose(1,2))
        # 进行softmax归一化
        dot = F.softmax(dot, dim=2)
        
        out = torch.bmm(dot, values).view(b, h, t, k)
        
        # swap h, t back, unify heads
        out = out.transpose(1, 2).contiguous().view(b, t, h*k)
        return self.unifyheads(out)
    
class TransformerBlock(nn.Module):
    def __init__(self, k, heads):
        super().__init__()

        self.attention = SelfAttention(k, heads = heads)

        self.norm1 = nn.LayerNorm(k)
        self.norm2 = nn.LayerNorm(k)

        self.mlp = nn.Sequential(
            nn.Linear(k, 4*k),
            nn.ReLU(),
            nn.Linear(4*k, k)
        )

    def forward(self, x):
        # 先做self-attention
        attended = self.attention(x)
        # 再做layer norm
        x = self.norm1(attended + x)

        # feedforward和layer norm
        feedforward = self.mlp(x)
        return self.norm2(feedforward + x)
    
class Transformer(nn.Module):
    def __init__(self, k, heads, depth):
    #def __init__(self, k, heads, depth, seq_length):
        super().__init__()
        # 引入embedding层
        #self.pos_emb = nn.Embedding(seq_length, k)

        # The sequence of transformer blocks that does all the 
        # heavy lifting
        tblocks = []
        for i in range(depth):
            tblocks.append(TransformerBlock(k=k, heads=heads))
        self.tblocks = nn.Sequential(*tblocks)

    def forward(self, x):
        """
        :param x: A (b, t) tensor of integer values representing 
                  words (in some predetermined vocabulary).
        :return: A (b, c) tensor of log-probabilities over the 
                 classes (where c is the nr. of classes).
        """
        b, t, k = x.size()

        # 产生 position embeddings
#         positions = torch.arange(t)
#         positions = self.pos_emb(positions)[None, :, :].expand(b, t, k)

#         x = x + positions
        x = self.tblocks(x)

        return x

class Transformer_Res2Net_Block(nn.Module):
    
    def __init__(self,channel=512,heads=4,depth=1,k=3,d=2,s=8,bottleneck=128):
        super(Transformer_Res2Net_Block,self).__init__()
        self.att = Transformer(k=channel, heads=heads, depth=depth)
        self.res2net = SE_Res2Block(k=k,d=d,s=s,channel=channel,bottleneck=bottleneck)
        
    def forward(self,x):
        x = self.att(x.transpose(1,2))
        x = self.res2net(x.transpose(1,2))
        return x

class Transformer_Res2Net(nn.Module):
    
    def __init__(self,channels=512,embedding_size=128):
        super(Transformer_Res2Net,self).__init__()
        self.conv1 = nn.Conv1d(80,channels,kernel_size=5,dilation=1)
        self.bn1 = nn.BatchNorm1d(channels)
        self.backbone1 = Transformer_Res2Net_Block(channel=channels,heads=4,depth=1,k=3,d=2,s=8,bottleneck=128)
        self.backbone2 = Transformer_Res2Net_Block(channel=channels,heads=4,depth=1,k=3,d=3,s=8,bottleneck=128)
        self.backbone3 = Transformer_Res2Net_Block(channel=channels,heads=4,depth=1,k=3,d=4,s=8,bottleneck=128)
        self.pool = StatsPool()
        self.bottleneck = nn.Linear(512*2, embedding_size)
    def forward(self,x):
        x = F.relu(self.bn1(self.conv1(x.transpose(1,2))))
        y_1 = self.backbone1(x)
        y_2 = self.backbone2(y_1) + y_1
        y_3 = self.backbone3(y_2) + y_2
        y = self.pool(y_3)
        
        return self.bottleneck(y)
    
    
##########################

class Attention_ResNet34StatsPool(nn.Module):

    def __init__(self, in_planes, embedding_size, dropout=0,heads=4,depth=1, **kwargs):

        super(Attention_ResNet34StatsPool, self).__init__()
        self.att = Transformer(80, heads=heads, depth=depth)
        self.front = ResNet34(in_planes,in_ch=2, **kwargs)
        self.pool = StatsPool()
        self.bottleneck = nn.Linear(in_planes*8*2, embedding_size)
        self.drop = nn.Dropout(dropout) if dropout else None
        
    def forward(self, x):
        att_x = self.att(x)
        x_in = torch.cat((att_x.unsqueeze(dim=1),x.unsqueeze(dim=1)), 1)
        x = self.front(x_in)
        x = self.pool(x)
        x = self.bottleneck(x)
        if self.drop:
            x = self.drop(x)
        return x
    
####################################################################
##### Trans_ECAPA_TDNN #######################################
####################################################################

class SelfAttention_Harvard(nn.Module):
    def __init__(self, k, heads = 8):
        super().__init__()
        self.k, self.heads = k, heads
        
        # input 维度为k（embedding结果），map成一个k*heads维度的矩阵
        self.tokeys    = nn.Linear(k, k, bias = False)
        self.toqueries = nn.Linear(k, k, bias = False)
        self.tovalues  = nn.Linear(k, k, bias = False)
        # 在通过线性转换把维度压缩到 k
        self.unifyheads = nn.Linear(k, k)
    
    def forward(self, x):
        b, t, k = x.size()
        h = self.heads

        queries = self.toqueries(x).view(b, t, h, k//h)
        keys    = self.tokeys(x).view(b, t, h, k//h)
        values  = self.tovalues(x).view(b, t, h, k//h)
        
        # 把 head 压缩进 batch的dimension
        queries = queries.transpose(1, 2).contiguous().view(b * h, t, k//h)
        keys    = keys.transpose(1, 2).contiguous().view(b * h, t, k//h)
        values  = values.transpose(1, 2).contiguous().view(b * h, t, k//h)
        
        # 这等效于对点积进行normalize
        queries = queries / (k//h ** (1/4))
        keys = keys / (k//h ** (1/4))
        # 矩阵相乘
        dot  = torch.bmm(queries, keys.transpose(1,2))
        # 进行softmax归一化
        dot = F.softmax(dot, dim=2)
        
        out = torch.bmm(dot, values).view(b, h, t, k//h)
        
        # swap h, t back, unify heads
        out = out.transpose(1, 2).contiguous().view(b, t, k)
        return self.unifyheads(out)
    
class TransformerBlock_Harvard(nn.Module):
    def __init__(self, k, heads):
        super().__init__()

        self.attention = SelfAttention_Harvard(k, heads = heads)

        self.norm1 = nn.LayerNorm(k)
        self.norm2 = nn.LayerNorm(k)

        self.mlp = nn.Sequential(
            nn.Linear(k, 4*k),
            nn.ReLU(),
            nn.Linear(4*k, k)
        )

    def forward(self, x):
        # 先做self-attention
        attended = self.attention(x)
        # 再做layer norm
        x = self.norm1(attended + x)

        # feedforward和layer norm
        feedforward = self.mlp(x)
        return self.norm2(feedforward + x)
    
class Transformer_Harvard(nn.Module):
    def __init__(self, k, heads, depth):
    #def __init__(self, k, heads, depth, seq_length):
        super().__init__()
        # 引入embedding层
        #self.pos_emb = nn.Embedding(seq_length, k)

        # The sequence of transformer blocks that does all the 
        # heavy lifting
        tblocks = []
        for i in range(depth):
            tblocks.append(TransformerBlock_Harvard(k=k, heads=heads))
        self.tblocks = nn.Sequential(*tblocks)

    def forward(self, x):
        """
        :param x: A (b, t) tensor of integer values representing 
                  words (in some predetermined vocabulary).
        :return: A (b, c) tensor of log-probabilities over the 
                 classes (where c is the nr. of classes).
        """
        b, t, k = x.size()

        # 产生 position embeddings
#         positions = torch.arange(t)
#         positions = self.pos_emb(positions)[None, :, :].expand(b, t, k)

#         x = x + positions
        x = self.tblocks(x)

        return x
    
class SE_TransBlock(nn.Module):
    
    def __init__(self,channel=512,bottleneck=128):
        super(SE_TransBlock,self).__init__()
            
        self.channel = channel
        self.bottleneck = bottleneck
        
        self.conv1 = nn.Conv1d(self.channel,self.channel,kernel_size=1,dilation=1)
        self.bn1 = nn.BatchNorm1d(self.channel)
        
        self.transformer = Transformer(self.channel, heads=4, depth=1)
        
        self.conv3 = nn.Conv1d(self.channel,self.channel,kernel_size=1,dilation=1)
        self.bn3 = nn.BatchNorm1d(self.channel)
        
        self.fc1 = nn.Linear(self.channel,self.bottleneck,bias=True)
        self.fc2 = nn.Linear(self.bottleneck,self.channel,bias=True)
        
    def forward(self,x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.transformer(out.transpose(1,2))
        out = F.relu(self.bn3(self.conv3(out.transpose(1,2))))
        out_mean = torch.mean(out,dim=2)
        s_v = torch.sigmoid(self.fc2(F.relu(self.fc1(out_mean))))
        out = out * s_v.unsqueeze(-1)
        out += residual
        #out = F.relu(out)
        return out
    
class Trans_ECAPA_TDNN(nn.Module):
    
    def __init__(self,in_dim=80,hidden_dim=512,bottleneck=128,embedding_size=128):
        
        super(Trans_ECAPA_TDNN,self).__init__()
        
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.bottleneck = bottleneck
        self.embedding_size = embedding_size
        
        self.conv1 = nn.Conv1d(in_dim,hidden_dim,kernel_size=5,dilation=1)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.block1 = SE_TransBlock(channel=self.hidden_dim,bottleneck=self.bottleneck)
        self.block2 = SE_TransBlock(channel=self.hidden_dim,bottleneck=self.bottleneck)
        self.block3 = SE_TransBlock(channel=self.hidden_dim,bottleneck=self.bottleneck)
        self.conv2 = nn.Conv1d(self.hidden_dim*3,self.hidden_dim*3,kernel_size=1,dilation=1)
        
        self.ASP = Attentive_Statictics_Pooling(channel=self.hidden_dim*3,R_dim_self_att=self.bottleneck)
        self.bn2 = nn.BatchNorm1d(self.hidden_dim*3*2)
        
        self.fc = nn.Linear(self.hidden_dim*3*2,self.embedding_size)
        self.bn3 = nn.BatchNorm1d(self.embedding_size)
        
    def forward(self,x):
        x = x.transpose(1,2)
        y = F.relu(self.bn1(self.conv1(x)))
        y_1 = self.block1(y)
        y_2 = self.block2(y_1)
        y_3 = self.block3(y_2)
        out = torch.cat((y_1, y_2,y_3), 1)
        out = F.relu(self.conv2(out))
        out = self.bn2(self.ASP(out.transpose(1,2)))
        out = self.bn3(self.fc(out))
        return out
    
#######################
### SE_Res_TDNN #######
#######################
class SE_Res_TDNN_Layer(nn.Module):
    def __init__(self, input_dim=512, output_dim=512, context_size=5, stride=1, dilation=1,bottleneck_dim=128):
        super(SE_Res_TDNN_Layer, self).__init__()
        self.tdnn = TDNNLayer(input_dim=input_dim,output_dim=output_dim,context_size=context_size, stride=stride, dilation=dilation, padding=dilation)
        self.fc1 = nn.Linear(output_dim,bottleneck_dim,bias=True)
        self.fc2 = nn.Linear(bottleneck_dim,output_dim,bias=True)
        self.nonlinearity = nonLinearAct()
    
    def forward(self, x):
        '''
        size (batch, input_features, seq_len)
        '''
        res_x = x
        out = self.tdnn(x)
        out_mean = torch.mean(out,dim=2)
        s_v = torch.sigmoid(self.fc2(self.nonlinearity(self.fc1(out_mean))))
        out = out * s_v.unsqueeze(-1)
        out += res_x
        return out  
    
class SE_Res_TDNN(nn.Module):
    def __init__(self, input_dim=80, output_dim=512,embedding_size=128,bottleneck_dim=128):
        super(SE_Res_TDNN, self).__init__()
        self.output_dim = output_dim
        self.bottleneck_dim = bottleneck_dim
        self.embedding_size = embedding_size
        self.tdnn1 = TDNNLayer(input_dim=input_dim,output_dim=output_dim,context_size=5, stride=1, dilation=1, padding=0)
        self.tdnn2 = SE_Res_TDNN_Layer(input_dim=self.output_dim, output_dim=self.output_dim, context_size=3, stride=1, dilation=2,bottleneck_dim=128)
        self.tdnn3 = SE_Res_TDNN_Layer(input_dim=self.output_dim, output_dim=self.output_dim, context_size=3, stride=1, dilation=3,bottleneck_dim=128)
        self.tdnn4 = SE_Res_TDNN_Layer(input_dim=self.output_dim, output_dim=self.output_dim, context_size=3, stride=1, dilation=4,bottleneck_dim=128)
        self.conv2 = nn.Conv1d(self.output_dim,self.output_dim,kernel_size=1,dilation=1)
        self.nonlinearity = nonLinearAct()
        self.ASP = Attentive_Statictics_Pooling(channel=self.output_dim,R_dim_self_att=self.bottleneck_dim)  
        self.bn2 = nn.BatchNorm1d(self.output_dim*2)
        
        self.fc = nn.Linear(self.output_dim*2,self.embedding_size)
        self.bn3 = nn.BatchNorm1d(self.embedding_size)
        
    def forward(self,x):
        out = self.tdnn1(x.transpose(1, 2))
        out = self.tdnn2(out)
        out = self.tdnn3(out)
        out = self.tdnn4(out)
        out = self.nonlinearity(self.conv2(out))
        out = self.bn2(self.ASP(out.transpose(1,2)))
        out = self.bn3(self.fc(out))
        return out
    
class Extend_SE_Res_TDNN(nn.Module):
    def __init__(self, input_dim=80, output_dim=512,embedding_size=128,bottleneck_dim=128):
        super(Extend_SE_Res_TDNN, self).__init__()
        self.output_dim = output_dim
        self.bottleneck_dim = bottleneck_dim
        self.embedding_size = embedding_size
        self.tdnn1 = TDNNLayer(input_dim=input_dim,output_dim=output_dim,context_size=5, stride=1, dilation=1, padding=0)
        self.tdnn2 = SE_Res_TDNN_Layer(input_dim=self.output_dim, output_dim=self.output_dim, context_size=3, stride=1, dilation=2,bottleneck_dim=self.bottleneck_dim)
        self.tdnn3 = SE_Res_TDNN_Layer(input_dim=self.output_dim, output_dim=self.output_dim, context_size=3, stride=1, dilation=1,bottleneck_dim=self.bottleneck_dim)
        self.tdnn4 = SE_Res_TDNN_Layer(input_dim=self.output_dim, output_dim=self.output_dim, context_size=3, stride=1, dilation=3,bottleneck_dim=self.bottleneck_dim)
        self.tdnn5 = SE_Res_TDNN_Layer(input_dim=self.output_dim, output_dim=self.output_dim, context_size=3, stride=1, dilation=1,bottleneck_dim=self.bottleneck_dim)
        self.tdnn6 = SE_Res_TDNN_Layer(input_dim=self.output_dim, output_dim=self.output_dim, context_size=3, stride=1, dilation=4,bottleneck_dim=self.bottleneck_dim)
        self.tdnn7 = SE_Res_TDNN_Layer(input_dim=self.output_dim, output_dim=self.output_dim, context_size=3, stride=1, dilation=1,bottleneck_dim=self.bottleneck_dim)
        self.tdnn8 = SE_Res_TDNN_Layer(input_dim=self.output_dim, output_dim=self.output_dim, context_size=3, stride=1, dilation=1,bottleneck_dim=self.bottleneck_dim)
        self.conv2 = nn.Conv1d(self.output_dim,self.output_dim,kernel_size=1,dilation=1)
        self.tdnn_list = nn.Sequential(self.tdnn1, self.tdnn2, self.tdnn3, self.tdnn4, self.tdnn5, self.tdnn6, self.tdnn7, self.tdnn8)
        self.nonlinearity = nonLinearAct()
        self.ASP = Attentive_Statictics_Pooling(channel=self.output_dim,R_dim_self_att=self.bottleneck_dim)  
        self.bn2 = nn.BatchNorm1d(self.output_dim*2)
        
        self.fc = nn.Linear(self.output_dim*2,self.embedding_size)
        self.bn3 = nn.BatchNorm1d(self.embedding_size)
        
    def forward(self,x):
        out = self.tdnn_list(x.transpose(1, 2))
        out = self.nonlinearity(self.conv2(out))
        out = self.bn2(self.ASP(out.transpose(1,2)))
        out = self.bn3(self.fc(out))
        return out
    
    
######################
### ResBlock_TDNN ###
######################


class ResBlock_TDNN(nn.Module):
    def __init__(self, in_planes, embedding_size=128, in_ch=1, is1d=False,**kwargs):
        super(ResBlock_TDNN, self).__init__()
        if is1d:
            self.NormLayer = nn.BatchNorm1d
            self.ConvLayer = nn.Conv1d
        else:
            self.NormLayer = nn.BatchNorm2d
            self.ConvLayer = nn.Conv2d

        self.in_planes = in_planes

        self.conv1 = self.ConvLayer(in_ch, in_planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = self.NormLayer(in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(BasicBlock_resnet34, in_planes, int(3), stride=1)
        self.conv2 = nn.Conv2d(in_planes, 1, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(80)
        
        self.tdnn = BIG_ETDNNStatsPool(80, embedding_size, dropout=0.0,hidden_features=1024)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.ConvLayer, self.NormLayer, self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x.unsqueeze(dim=1))))
        x = self.layer1(x)
        x = self.relu(self.bn2(self.conv2(x).squeeze().transpose(1,2)))
        x = self.tdnn(x.transpose(1,2))
        return x
    
class Stand_ResBlock_TDNN(nn.Module):
    def __init__(self, in_planes, embedding_size=128, in_ch=1, is1d=False,**kwargs):
        super(Stand_ResBlock_TDNN, self).__init__()
        if is1d:
            self.NormLayer = nn.BatchNorm1d
            self.ConvLayer = nn.Conv1d
        else:
            self.NormLayer = nn.BatchNorm2d
            self.ConvLayer = nn.Conv2d

        self.in_planes = in_planes

        self.conv1 = self.ConvLayer(in_ch, in_planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = self.NormLayer(in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(BasicBlock_resnet34, in_planes, int(3), stride=1)
        self.conv2 = nn.Conv2d(in_planes, 1, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(80)
        
        self.tdnn = Standard_TDNN_GE2E(80, embedding_size, dropout=0.0,hidden_features=1024)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.ConvLayer, self.NormLayer, self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x.unsqueeze(dim=1))))
        x = self.layer1(x)
        x = self.relu(self.bn2(self.conv2(x).squeeze().transpose(1,2)))
        x = self.tdnn(x.transpose(1,2))
        return x
    
##########################
### Gradient Reversal ###    
#########################

class grl_func(torch.autograd.Function):
    def __init__(self):
        super(grl_func, self).__init__()

    @ staticmethod
    def forward(ctx, x, lambda_):
        ctx.save_for_backward(lambda_)
        return x.view_as(x)

    @ staticmethod
    def backward(ctx, grad_output):
        lambda_, = ctx.saved_variables
        grad_input = grad_output.clone()
        return - lambda_ * grad_input, None


class GRL(nn.Module):
    def __init__(self, lambda_=0.):
        super(GRL, self).__init__()
        self.lambda_ = torch.tensor(lambda_)

    def set_lambda(self, lambda_):
        self.lambda_ = torch.tensor(lambda_)

    def forward(self, x):
        return grl_func.apply(x, self.lambda_)
    
