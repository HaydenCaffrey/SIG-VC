import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def get_casual_padding1d():
    pass

def get_casual_padding2d():
    pass

class cPReLU(nn.Module):

    def __init__(self, complex_axis=1):
        super(cPReLU,self).__init__()
        self.r_prelu = nn.PReLU()        
        self.i_prelu = nn.PReLU()
        self.complex_axis = complex_axis


    def forward(self, inputs):
        real, imag = torch.chunk(inputs, 2,self.complex_axis)
        real = self.r_prelu(real)
        imag = self.i_prelu(imag)
        return torch.cat([real,imag],self.complex_axis)

class NavieComplexLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, projection_dim=None, bidirectional=False, batch_first=False):
        super(NavieComplexLSTM, self).__init__()

        self.input_dim = input_size//2
        self.rnn_units = hidden_size//2
        self.real_lstm = nn.LSTM(self.input_dim, self.rnn_units, num_layers=1, bidirectional=bidirectional, batch_first=False)
        self.imag_lstm = nn.LSTM(self.input_dim, self.rnn_units, num_layers=1, bidirectional=bidirectional, batch_first=False)
        if bidirectional:
            bidirectional=2
        else:
            bidirectional=1
        if projection_dim is not None:
            self.projection_dim = projection_dim//2 
            self.r_trans = nn.Linear(self.rnn_units*bidirectional, self.projection_dim)
            self.i_trans = nn.Linear(self.rnn_units*bidirectional, self.projection_dim)
        else:
            self.projection_dim = None

    def forward(self, inputs):
        if isinstance(inputs,list):
            real, imag = inputs 
        elif isinstance(inputs, torch.Tensor):
            real, imag = torch.chunk(inputs,-1)
        r2r_out = self.real_lstm(real)[0]
        r2i_out = self.imag_lstm(real)[0]
        i2r_out = self.real_lstm(imag)[0]
        i2i_out = self.imag_lstm(imag)[0]
        real_out = r2r_out - i2i_out
        imag_out = i2r_out + r2i_out 
        if self.projection_dim is not None:
            real_out = self.r_trans(real_out)
            imag_out = self.i_trans(imag_out)
        #print(real_out.shape,imag_out.shape)
        return [real_out, imag_out]
    
    def flatten_parameters(self):
        self.imag_lstm.flatten_parameters()
        self.real_lstm.flatten_parameters()

def complex_cat(inputs, axis):
    
    real, imag = [],[]
    for idx, data in enumerate(inputs):
        r, i = torch.chunk(data,2,axis)
        real.append(r)
        imag.append(i)
    real = torch.cat(real,axis)
    imag = torch.cat(imag,axis)
    outputs = torch.cat([real, imag],axis)
    return outputs

class ComplexConv2d(nn.Module):

    def __init__(
                    self,
                    in_channels,
                    out_channels,
                    kernel_size=(1,1),
                    stride=(1,1),
                    padding=(0,0),
                    dilation=1,
                    groups = 1,
                    causal=True, 
                    complex_axis=1,
                ):
        '''
            in_channels: real+imag
            out_channels: real+imag 
            kernel_size : input [B,C,D,T] kernel size in [D,T]
            padding : input [B,C,D,T] padding in [D,T]
            causal: if causal, will padding time dimension's left side,
                    otherwise both

        '''
        super(ComplexConv2d, self).__init__()
        self.in_channels = in_channels//2
        self.out_channels = out_channels//2
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.causal = causal
        self.groups = groups
        self.dilation = dilation
        self.complex_axis=complex_axis        
        self.real_conv = nn.Conv2d(self.in_channels, self.out_channels, kernel_size, self.stride,padding=[self.padding[0],0],dilation=self.dilation, groups=self.groups)  
        self.imag_conv = nn.Conv2d(self.in_channels, self.out_channels, kernel_size, self.stride,padding=[self.padding[0],0],dilation=self.dilation, groups=self.groups)  
        
        nn.init.normal_(self.real_conv.weight.data,std=0.05)
        nn.init.normal_(self.imag_conv.weight.data,std=0.05) 
        nn.init.constant_(self.real_conv.bias,0.) 
        nn.init.constant_(self.imag_conv.bias,0.) 
        

    def forward(self,inputs):
        if self.padding[1] != 0 and self.causal:
            inputs = F.pad(inputs,[self.padding[1], 0,0,0]) 
        else:
            inputs = F.pad(inputs,[self.padding[1], self.padding[1],0,0]) 

        if self.complex_axis == 0:
            real = self.real_conv(inputs)
            imag = self.imag_conv(inputs)
            real2real,imag2real = torch.chunk(real,2, self.complex_axis)
            real2imag,imag2imag = torch.chunk(imag,2, self.complex_axis)

        else:
            if isinstance(inputs, torch.Tensor):
                real,imag = torch.chunk(inputs, 2, self.complex_axis)
        
            real2real = self.real_conv(real,)
            imag2imag = self.imag_conv(imag,)
        
            real2imag = self.imag_conv(real)
            imag2real = self.real_conv(imag)

        real = real2real - imag2imag
        imag = real2imag + imag2real
        out = torch.cat([real, imag], self.complex_axis)
        
        return out


class ComplexConvTranspose2d(nn.Module):

    def __init__(
                    self,
                    in_channels,
                    out_channels,
                    kernel_size=(1,1),
                    stride=(1,1),
                    padding=(0,0),
                    output_padding=(0,0),
                    causal=False,
                    complex_axis=1,
                    groups=1
                ):
        '''
            in_channels: real+imag
            out_channels: real+imag
        '''
        super(ComplexConvTranspose2d, self).__init__()
        self.in_channels = in_channels//2
        self.out_channels = out_channels//2
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding=output_padding 
        self.groups = groups 
        
        self.real_conv = nn.ConvTranspose2d(self.in_channels, self.out_channels,kernel_size, self.stride,padding=self.padding,output_padding=output_padding, groups=self.groups)  
        self.imag_conv = nn.ConvTranspose2d(self.in_channels, self.out_channels,kernel_size, self.stride,padding=self.padding,output_padding=output_padding, groups=self.groups)  
        self.complex_axis=complex_axis        

        nn.init.normal_(self.real_conv.weight,std=0.05)
        nn.init.normal_(self.imag_conv.weight,std=0.05) 
        nn.init.constant_(self.real_conv.bias,0.) 
        nn.init.constant_(self.imag_conv.bias,0.) 

    def forward(self,inputs):
         
        if isinstance(inputs, torch.Tensor):
            real,imag = torch.chunk(inputs, 2, self.complex_axis)
        elif isinstance(inputs, tuple) or isinstance(inputs, list):
            real = inputs[0]
            imag = inputs[1]
        if self.complex_axis == 0:
            real = self.real_conv(inputs)
            imag = self.imag_conv(inputs)
            real2real,imag2real = torch.chunk(real,2, self.complex_axis)
            real2imag,imag2imag = torch.chunk(imag,2, self.complex_axis)

        else:
            if isinstance(inputs, torch.Tensor):
                real,imag = torch.chunk(inputs, 2, self.complex_axis)
        
            real2real = self.real_conv(real,)
            imag2imag = self.imag_conv(imag,)
        
            real2imag = self.imag_conv(real)
            imag2real = self.real_conv(imag)
        
        real = real2real - imag2imag
        imag = real2imag + imag2real
        out = torch.cat([real, imag], self.complex_axis)
        
        return out



# Source: https://github.com/ChihebTrabelsi/deep_complex_networks/tree/pytorch 
# from https://github.com/IMLHF/SE_DCUNet/blob/f28bf1661121c8901ad38149ea827693f1830715/models/layers/complexnn.py#L55

class ComplexBatchNorm(torch.nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True,
            track_running_stats=True, complex_axis=1):
        super(ComplexBatchNorm, self).__init__()
        self.num_features        = num_features//2
        self.eps                 = eps
        self.momentum            = momentum
        self.affine              = affine
        self.track_running_stats = track_running_stats 
        
        self.complex_axis = complex_axis

        if self.affine:
            self.Wrr = torch.nn.Parameter(torch.Tensor(self.num_features))
            self.Wri = torch.nn.Parameter(torch.Tensor(self.num_features))
            self.Wii = torch.nn.Parameter(torch.Tensor(self.num_features))
            self.Br  = torch.nn.Parameter(torch.Tensor(self.num_features))
            self.Bi  = torch.nn.Parameter(torch.Tensor(self.num_features))
        else:
            self.register_parameter('Wrr', None)
            self.register_parameter('Wri', None)
            self.register_parameter('Wii', None)
            self.register_parameter('Br',  None)
            self.register_parameter('Bi',  None)
        
        if self.track_running_stats:
            self.register_buffer('RMr',  torch.zeros(self.num_features))
            self.register_buffer('RMi',  torch.zeros(self.num_features))
            self.register_buffer('RVrr', torch.ones (self.num_features))
            self.register_buffer('RVri', torch.zeros(self.num_features))
            self.register_buffer('RVii', torch.ones (self.num_features))
            self.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long))
        else:
            self.register_parameter('RMr',                 None)
            self.register_parameter('RMi',                 None)
            self.register_parameter('RVrr',                None)
            self.register_parameter('RVri',                None)
            self.register_parameter('RVii',                None)
            self.register_parameter('num_batches_tracked', None)
        self.reset_parameters()

    def reset_running_stats(self):
        if self.track_running_stats:
            self.RMr.zero_()
            self.RMi.zero_()
            self.RVrr.fill_(1)
            self.RVri.zero_()
            self.RVii.fill_(1)
            self.num_batches_tracked.zero_()

    def reset_parameters(self):
        self.reset_running_stats()
        if self.affine:
            self.Br.data.zero_()
            self.Bi.data.zero_()
            self.Wrr.data.fill_(1)
            self.Wri.data.uniform_(-.9, +.9) # W will be positive-definite
            self.Wii.data.fill_(1)

    def _check_input_dim(self, xr, xi):
        assert(xr.shape == xi.shape)
        assert(xr.size(1) == self.num_features)

    def forward(self, inputs):
        #self._check_input_dim(xr, xi)
        
        xr, xi = torch.chunk(inputs,2, axis=self.complex_axis)
        exponential_average_factor = 0.0

        if self.training and self.track_running_stats:
            self.num_batches_tracked += 1
            if self.momentum is None:  # use cumulative moving average
                exponential_average_factor = 1.0 / self.num_batches_tracked.item()
            else:  # use exponential moving average
                exponential_average_factor = self.momentum

        #
        # NOTE: The precise meaning of the "training flag" is:
        #       True:  Normalize using batch   statistics, update running statistics
        #              if they are being collected.
        #       False: Normalize using running statistics, ignore batch   statistics.
        #
        training = self.training or not self.track_running_stats
        redux = [i for i in reversed(range(xr.dim())) if i!=1]
        vdim  = [1] * xr.dim()
        vdim[1] = xr.size(1)

        #
        # Mean M Computation and Centering
        #
        # Includes running mean update if training and running.
        #
        if training:
            Mr, Mi = xr, xi
            for d in redux:
                Mr = Mr.mean(d, keepdim=True)
                Mi = Mi.mean(d, keepdim=True)
            if self.track_running_stats:
                self.RMr.lerp_(Mr.squeeze(), exponential_average_factor)
                self.RMi.lerp_(Mi.squeeze(), exponential_average_factor)
        else:
            Mr = self.RMr.view(vdim)
            Mi = self.RMi.view(vdim)
        xr, xi = xr-Mr, xi-Mi

        #
        # Variance Matrix V Computation
        #
        # Includes epsilon numerical stabilizer/Tikhonov regularizer.
        # Includes running variance update if training and running.
        #
        if training:
            Vrr = xr * xr
            Vri = xr * xi
            Vii = xi * xi
            for d in redux:
                Vrr = Vrr.mean(d, keepdim=True)
                Vri = Vri.mean(d, keepdim=True)
                Vii = Vii.mean(d, keepdim=True)
            if self.track_running_stats:
                self.RVrr.lerp_(Vrr.squeeze(), exponential_average_factor)
                self.RVri.lerp_(Vri.squeeze(), exponential_average_factor)
                self.RVii.lerp_(Vii.squeeze(), exponential_average_factor)
        else:
            Vrr = self.RVrr.view(vdim)
            Vri = self.RVri.view(vdim)
            Vii = self.RVii.view(vdim)
        Vrr   = Vrr + self.eps
        Vri   = Vri
        Vii   = Vii + self.eps

        #
        # Matrix Inverse Square Root U = V^-0.5
        #
        # sqrt of a 2x2 matrix,
        # - https://en.wikipedia.org/wiki/Square_root_of_a_2_by_2_matrix
        tau   = Vrr + Vii
        delta = torch.addcmul(Vrr * Vii, -1, Vri, Vri)
        s     = delta.sqrt()
        t     = (tau + 2*s).sqrt()

        # matrix inverse, http://mathworld.wolfram.com/MatrixInverse.html
        rst   = (s * t).reciprocal()
        Urr   = (s + Vii) * rst
        Uii   = (s + Vrr) * rst
        Uri   = (  - Vri) * rst

        #
        # Optionally left-multiply U by affine weights W to produce combined
        # weights Z, left-multiply the inputs by Z, then optionally bias them.
        #
        # y = Zx + B
        # y = WUx + B
        # y = [Wrr Wri][Urr Uri] [xr] + [Br]
        #     [Wir Wii][Uir Uii] [xi]   [Bi]
        #
        if self.affine:
            Wrr, Wri, Wii = self.Wrr.view(vdim), self.Wri.view(vdim), self.Wii.view(vdim)
            Zrr = (Wrr * Urr) + (Wri * Uri)
            Zri = (Wrr * Uri) + (Wri * Uii)
            Zir = (Wri * Urr) + (Wii * Uri)
            Zii = (Wri * Uri) + (Wii * Uii)
        else:
            Zrr, Zri, Zir, Zii = Urr, Uri, Uri, Uii

        yr = (Zrr * xr) + (Zri * xi)
        yi = (Zir * xr) + (Zii * xi)

        if self.affine:
            yr = yr + self.Br.view(vdim)
            yi = yi + self.Bi.view(vdim)

        outputs = torch.cat([yr, yi], self.complex_axis)
        return outputs

    def extra_repr(self):
        return '{num_features}, eps={eps}, momentum={momentum}, affine={affine}, ' \
                'track_running_stats={track_running_stats}'.format(**self.__dict__) 

def complex_cat(inputs, axis):
    
    real, imag = [],[]
    for idx, data in enumerate(inputs):
        r, i = torch.chunk(data,2,axis)
        real.append(r)
        imag.append(i)
    real = torch.cat(real,axis)
    imag = torch.cat(imag,axis)
    outputs = torch.cat([real, imag],axis)
    return outputs




import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from scipy.signal import get_window


def init_kernels(win_len, win_inc, fft_len, win_type=None, invers=False):
    if win_type == 'None' or win_type is None:
        window = np.ones(win_len)
    else:
        window = get_window(win_type, win_len, fftbins=True)#**0.5
    
    N = fft_len
    fourier_basis = np.fft.rfft(np.eye(N))[:win_len]
    real_kernel = np.real(fourier_basis)
    imag_kernel = np.imag(fourier_basis)
    kernel = np.concatenate([real_kernel, imag_kernel], 1).T
    
    if invers :
        kernel = np.linalg.pinv(kernel).T 

    kernel = kernel*window
    kernel = kernel[:, None, :]
    return torch.from_numpy(kernel.astype(np.float32)), torch.from_numpy(window[None,:,None].astype(np.float32))


class ConvSTFT(nn.Module):

    def __init__(self, win_len, win_inc, fft_len=None, win_type='hamming', feature_type='real', fix=True):
        super(ConvSTFT, self).__init__() 
        
        if fft_len == None:
            self.fft_len = np.int(2**np.ceil(np.log2(win_len)))
        else:
            self.fft_len = fft_len
        
        kernel, _ = init_kernels(win_len, win_inc, self.fft_len, win_type)
        #self.weight = nn.Parameter(kernel, requires_grad=(not fix))
        self.register_buffer('weight', kernel)
        self.feature_type = feature_type
        self.stride = win_inc
        self.win_len = win_len
        self.dim = self.fft_len

    def forward(self, inputs):
        if inputs.dim() == 2:
            inputs = torch.unsqueeze(inputs, 1)
        inputs = F.pad(inputs,[self.win_len-self.stride, self.win_len-self.stride])
        outputs = F.conv1d(inputs, self.weight, stride=self.stride)
         
        if self.feature_type == 'complex':
            return outputs
        else:
            dim = self.dim//2+1
            real = outputs[:, :dim, :]
            imag = outputs[:, dim:, :]
            mags = torch.sqrt(real**2+imag**2)
            phase = torch.atan2(imag, real)
            return mags, phase

class ConviSTFT(nn.Module):

    def __init__(self, win_len, win_inc, fft_len=None, win_type='hamming', feature_type='real', fix=True):
        super(ConviSTFT, self).__init__() 
        if fft_len == None:
            self.fft_len = np.int(2**np.ceil(np.log2(win_len)))
        else:
            self.fft_len = fft_len
        kernel, window = init_kernels(win_len, win_inc, self.fft_len, win_type, invers=True)
        #self.weight = nn.Parameter(kernel, requires_grad=(not fix))
        self.register_buffer('weight', kernel)
        self.feature_type = feature_type
        self.win_type = win_type
        self.win_len = win_len
        self.stride = win_inc
        self.stride = win_inc
        self.dim = self.fft_len
        self.register_buffer('window', window)
        self.register_buffer('enframe', torch.eye(win_len)[:,None,:])

    def forward(self, inputs, phase=None):
        """
        inputs : [B, N+2, T] (complex spec) or [B, N//2+1, T] (mags)
        phase: [B, N//2+1, T] (if not none)
        """ 

        if phase is not None:
            real = inputs*torch.cos(phase)
            imag = inputs*torch.sin(phase)
            inputs = torch.cat([real, imag], 1)
        outputs = F.conv_transpose1d(inputs, self.weight, stride=self.stride) 

        # this is from torch-stft: https://github.com/pseeth/torch-stft
        t = self.window.repeat(1,1,inputs.size(-1))**2
        coff = F.conv_transpose1d(t, self.enframe, stride=self.stride)
        outputs = outputs/(coff+1e-8)
        #outputs = torch.where(coff == 0, outputs, outputs/coff)
        outputs = outputs[...,self.win_len-self.stride:-(self.win_len-self.stride)]
        
        return outputs

def test_fft():
    torch.manual_seed(20)
    win_len = 320
    win_inc = 160 
    fft_len = 512
    inputs = torch.randn([1, 1, 16000*4])
    fft = ConvSTFT(win_len, win_inc, fft_len, win_type='hanning', feature_type='real')
    import librosa

    outputs1 = fft(inputs)[0]
    outputs1 = outputs1.numpy()[0]
    np_inputs = inputs.numpy().reshape([-1])
    librosa_stft = librosa.stft(np_inputs, win_length=win_len, n_fft=fft_len, hop_length=win_inc, center=False)
    print(np.mean((outputs1 - np.abs(librosa_stft))**2))


def test_ifft1():
    import soundfile as sf
    N = 400
    inc = 100
    fft_len=512
    torch.manual_seed(N)
    data = np.random.randn(16000*8)[None,None,:]
#    data = sf.read('../ori.wav')[0]
    inputs = data.reshape([1,1,-1])
    fft = ConvSTFT(N, inc, fft_len=fft_len, win_type='hanning', feature_type='complex')
    ifft = ConviSTFT(N, inc, fft_len=fft_len, win_type='hanning', feature_type='complex')
    inputs = torch.from_numpy(inputs.astype(np.float32))
    outputs1 = fft(inputs)
    print(outputs1.shape) 
    outputs2 = ifft(outputs1)
    sf.write('conv_stft.wav', outputs2.numpy()[0,0,:],16000)
    print('wav MSE', torch.mean(torch.abs(inputs[...,:outputs2.size(2)]-outputs2)**2))


def test_ifft2():
    N = 400
    inc = 100
    fft_len=512
    np.random.seed(20)
    torch.manual_seed(20)
    t = np.random.randn(16000*4)*0.001
    t = np.clip(t, -1, 1)
    #input = torch.randn([1,16000*4]) 
    input = torch.from_numpy(t[None,None,:].astype(np.float32))
    
    fft = ConvSTFT(N, inc, fft_len=fft_len, win_type='hanning', feature_type='complex')
    ifft = ConviSTFT(N, inc, fft_len=fft_len, win_type='hanning', feature_type='complex')
    
    out1 = fft(input)
    output = ifft(out1)
    print('random MSE', torch.mean(torch.abs(input-output)**2))
    import soundfile as sf
    sf.write('zero.wav', output[0,0].numpy(),16000)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, ConvLayer, NormLayer, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = ConvLayer(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = NormLayer(planes)
        self.conv2 = ConvLayer(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = NormLayer(planes)
        self.relu = nn.ReLU(inplace=True)
        
        self.downsample = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.downsample = nn.Sequential(
                ConvLayer(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                NormLayer(self.expansion*planes)
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.downsample(x)
        out = self.relu(out)
        return out    

class cBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, ConvLayer, NormLayer, in_planes, planes, stride=1):
        super(cBasicBlock, self).__init__()
        self.conv1 = ConvLayer(in_planes, planes, kernel_size=5, stride=(1, stride), padding=(2, 4))
        self.bn1 = NormLayer(planes)
        self.conv2 = ConvLayer(planes, planes, kernel_size=5, stride=(1, 1), padding=(2, 4))
        self.bn2 = NormLayer(planes)
        self.relu = nn.PReLU()
        
        self.downsample = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.downsample = nn.Sequential(
                ConvLayer(in_planes, self.expansion*planes, kernel_size=1, stride=(1, stride)),
                NormLayer(self.expansion*planes)
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.downsample(x)
        out = self.relu(out)
        return out

class cResNet(nn.Module):
    def __init__(self, in_planes, block, num_blocks, num_classes=10, in_ch=2, is1d=False, **kwargs):
        super(cResNet, self).__init__()
        self.N = 400
        self.inc = 100
        self.fft_len=512
        self.stft = ConvSTFT(self.N, self.inc, fft_len=self.fft_len, win_type='hanning', feature_type='complex')
        
        self.NormLayer = nn.BatchNorm2d
        self.ConvLayer = ComplexConv2d

        self.in_planes = in_planes

        self.conv1 = self.ConvLayer(in_ch, in_planes, kernel_size=5, stride=(2, 1), padding=(2, 1))
        self.bn1 = self.NormLayer(in_planes)
        self.relu = nn.PReLU()
        self.layer1 = self._make_layer(block, in_planes, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, in_planes*2, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, in_planes*4, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, in_planes*8, num_blocks[3], stride=2)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.ConvLayer, self.NormLayer, self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        specs = self.stft(x)
        real = specs[:,:self.fft_len//2+1]
        imag = specs[:,self.fft_len//2+1:]
        spec_mags = torch.sqrt(real**2+imag**2+1e-8)
        spec_mags = spec_mags
        spec_phase = torch.atan2(imag, real)
        spec_phase = spec_phase
        cspecs = torch.stack([real,imag],1)
        cspecs = cspecs[:,:,1:]
        x = self.relu(self.bn1(self.conv1(cspecs)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x
  
class convResNet(nn.Module):
    def __init__(self, in_planes, block, num_blocks, num_classes=10, in_ch=2, is1d=False, **kwargs):
        super(convResNet, self).__init__()
        self.N = 400
        self.inc = 100
        self.fft_len=512
        self.stft = ConvSTFT(self.N, self.inc, fft_len=self.fft_len, win_type='hanning', feature_type='complex')
        
        self.cNormLayer = nn.BatchNorm2d
        self.cConvLayer = ComplexConv2d
        
        self.NormLayer = nn.BatchNorm2d
        self.ConvLayer = nn.Conv2d

        self.in_planes = in_planes

        self.conv1 = self.cConvLayer(in_ch, in_planes*2, kernel_size=5, stride=(2, 1), padding=(2, 1))
#         self.conv1 = self.cConvLayer(in_ch, in_planes, kernel_size=5, stride=(2, 1), padding=(2, 1))
        self.bn1 = self.cNormLayer(in_planes*2)
        self.relu = nn.PReLU()
        
        self.layer1 = self._make_layer(block, in_planes, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, in_planes*2, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, in_planes*4, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, in_planes*8, num_blocks[3], stride=2)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.ConvLayer, self.NormLayer, self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        specs = self.stft(x)
        real = specs[:,:self.fft_len//2+1]
        imag = specs[:,self.fft_len//2+1:]
        spec_mags = torch.sqrt(real**2+imag**2+1e-8)
        spec_mags = spec_mags
        spec_phase = torch.atan2(imag, real)
        spec_phase = spec_phase
        cspecs = torch.stack([real,imag],1)
        x = cspecs[:,:,1:]
#         print(cspecs.shape)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = torch.log(torch.sqrt(x[:,:int(self.in_planes/8)]**2+x[:,int(self.in_planes/8):]**2+1e-8))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x

def cResNet18(in_planes, **kwargs):
    return cResNet(in_planes, cBasicBlock, [2,2,2,2], **kwargs)

def cResNet34(in_planes, **kwargs):
    return cResNet(in_planes, cBasicBlock, [3,4,6,3], **kwargs)

def convResNet34(in_planes, **kwargs):
    return convResNet(in_planes, BasicBlock, [3,4,6,3], **kwargs)

