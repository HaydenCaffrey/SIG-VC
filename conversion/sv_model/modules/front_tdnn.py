import torch, torch.nn as nn, torch.nn.functional as F

def nonLinearAct():
    return nn.LeakyReLU()


class TDNNLayer(nn.Module):
    def __init__(self, input_dim=23, output_dim=512, context_size=5, stride=1, dilation=1, padding=0):
        super(TDNNLayer, self).__init__()
        self.kernel = nn.Conv1d(input_dim, output_dim, context_size, stride=stride, padding=padding, dilation=dilation)
        self.nonlinearity = nonLinearAct()
        self.bn = nn.BatchNorm1d(output_dim)
        #self.drop = nn.Dropout(p=dropout_p)

    def forward(self, x):
        '''
        size (batch, input_features, seq_len)
        '''
        x = self.kernel(x)
        x = self.nonlinearity(x)
        #x = self.drop(x)
        x = self.bn(x)
        return x
    

class SOrthConv(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, padding_mode='zeros'):
        '''
        Conv1d with a method for stepping towards semi-orthongonality
        http://danielpovey.com/files/2018_interspeech_tdnnf.pdf
        '''
        super(SOrthConv, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, bias=False, padding_mode=padding_mode)
        self.reset_parameters()

    def forward(self, x):
        x = self.conv(x)
        return x

    def step_semi_orth(self):
        with torch.no_grad():
            M = self.get_semi_orth_weight(self.conv)
            self.conv.weight.copy_(M)

    def reset_parameters(self):
        # Standard dev of M init values is inverse of sqrt of num cols
        nn.init._no_grad_normal_(self.conv.weight, 0., self.get_M_shape(self.conv.weight)[1]**-0.5)

    def orth_error(self):
        return self.get_semi_orth_error(self.conv).item()

    @staticmethod
    def get_semi_orth_weight(conv1dlayer):
        # updates conv1 weight M using update rule to make it more semi orthogonal
        # based off ConstrainOrthonormalInternal in nnet-utils.cc in Kaldi src/nnet3
        # includes the tweaks related to slowing the update speed
        # only an implementation of the 'floating scale' case
        with torch.no_grad():
            update_speed = 0.125
            orig_shape = conv1dlayer.weight.shape
            # a conv weight differs slightly from TDNN formulation:
            # Conv weight: (out_filters, in_filters, kernel_width)
            # TDNN weight M is of shape: (in_dim, out_dim) or [rows, cols]
            # the in_dim of the TDNN weight is equivalent to in_filters * kernel_width of the Conv
            M = conv1dlayer.weight.reshape(
                orig_shape[0], orig_shape[1]*orig_shape[2]).T
            # M now has shape (in_dim[rows], out_dim[cols])
            mshape = M.shape
            if mshape[0] > mshape[1]:  # semi orthogonal constraint for rows > cols
                M = M.T
            P = torch.mm(M, M.T)
            PP = torch.mm(P, P.T)
            trace_P = torch.trace(P)
            trace_PP = torch.trace(PP)
            ratio = trace_PP * P.shape[0] / (trace_P * trace_P)

            # the following is the tweak to avoid divergence (more info in Kaldi)
            assert ratio > 0.99
            if ratio > 1.02:
                update_speed *= 0.5
                if ratio > 1.1:
                    update_speed *= 0.5

            scale2 = trace_PP/trace_P
            update = P - (torch.matrix_power(P, 0) * scale2)
            alpha = update_speed / scale2
            update = (-4.0 * alpha) * torch.mm(update, M)
            updated = M + update
            # updated has shape (cols, rows) if rows > cols, else has shape (rows, cols)
            # Transpose (or not) to shape (cols, rows) (IMPORTANT, s.t. correct dimensions are reshaped)
            # Then reshape to (cols, in_filters, kernel_width)
            return updated.reshape(*orig_shape) if mshape[0] > mshape[1] else updated.T.reshape(*orig_shape)

    @staticmethod
    def get_M_shape(conv_weight):
        orig_shape = conv_weight.shape
        return (orig_shape[1]*orig_shape[2], orig_shape[0])

    @staticmethod
    def get_semi_orth_error(conv1dlayer):
        with torch.no_grad():
            orig_shape = conv1dlayer.weight.shape
            M = conv1dlayer.weight.reshape(
                orig_shape[0], orig_shape[1]*orig_shape[2]).T
            mshape = M.shape
            if mshape[0] > mshape[1]:  # semi orthogonal constraint for rows > cols
                M = M.T
            P = torch.mm(M, M.T)
            PP = torch.mm(P, P.T)
            trace_P = torch.trace(P)
            trace_PP = torch.trace(PP)
            scale2 = torch.sqrt(trace_PP/trace_P) ** 2
            update = P - (torch.matrix_power(P, 0) * scale2)
            return torch.norm(update, p='fro')


class SharedDimScaleDropout(nn.Module):
    def __init__(self, alpha: float = 0.5, dim=2):
        '''
        Continuous scaled dropout that is const over chosen dim (usually across time)
        Multiplies inputs by random mask taken from Uniform([1 - 2\alpha, 1 + 2\alpha])
        '''
        super(SharedDimScaleDropout, self).__init__()
        if alpha > 0.5 or alpha < 0:
            raise ValueError("alpha must be between 0 and 0.5")
        self.alpha = alpha
        self.dim = dim
        self.register_buffer('mask', torch.tensor(0.))

    def forward(self, X):
        if self.training:
            if self.alpha != 0.:
                # sample mask from uniform dist with dim of length 1 in self.dim and then repeat to match size
                tied_mask_shape = list(X.shape)
                tied_mask_shape[self.dim] = 1
                repeats = [1 if i != self.dim else X.shape[self.dim] for i in range(len(X.shape))]
                return X * self.mask.repeat(tied_mask_shape).uniform_(1 - 2*self.alpha, 1 + 2*self.alpha).repeat(repeats)
                # expected value of dropout mask is 1 so no need to scale outputs like vanilla dropout
        return X


class FTDNNLayer(nn.Module):

    def __init__(self, in_dim, out_dim, bottleneck_dim, context_size=2, dilations=None, paddings=None, alpha=0.0):
        '''
        3 stage factorised TDNN http://danielpovey.com/files/2018_interspeech_tdnnf.pdf
        '''
        super(FTDNNLayer, self).__init__()
        paddings = [1, 1, 1] if not paddings else paddings
        dilations = [2, 2, 2] if not dilations else dilations
        assert len(paddings) == 3
        assert len(dilations) == 3
        self.factor1 = SOrthConv(in_dim, bottleneck_dim, context_size, padding=paddings[0], dilation=dilations[0])
        self.factor2 = SOrthConv(bottleneck_dim, bottleneck_dim, context_size, padding=paddings[1], dilation=dilations[1])
        self.factor3 = nn.Conv1d(bottleneck_dim, out_dim, context_size, padding=paddings[2], dilation=dilations[2], bias=False)
        self.nl = nonLinearAct()
        self.bn = nn.BatchNorm1d(out_dim)
        self.dropout = SharedDimScaleDropout(alpha=alpha, dim=2)

    def forward(self, x):
        ''' input (batch_size, in_dim, seq_len) '''
        x = self.factor1(x)
        x = self.factor2(x)
        x = self.factor3(x)
        x = self.nl(x)
        x = self.bn(x)
        x = self.dropout(x)
        return x

    def step_semi_orth(self):
        for layer in self.children():
            if isinstance(layer, SOrthConv):
                layer.step_semi_orth()

    def orth_error(self):
        orth_error = 0
        for layer in self.children():
            if isinstance(layer, SOrthConv):
                orth_error += layer.orth_error()
        return orth_error


class DenseReLU(nn.Module):

    def __init__(self, in_dim, out_dim):
        super(DenseReLU, self).__init__()
        self.fc = nn.Linear(in_dim, out_dim)
        self.bn = nn.BatchNorm1d(out_dim)
        self.nl = nonLinearAct()

    def forward(self, x):
        x = self.fc(x.transpose(1, 2)).transpose(1, 2)
        x = self.nl(x)
        x = self.bn(x)
        return x


class FTDNN(nn.Module):

    def __init__(self, in_dim=30):
        '''
        The FTDNN architecture from
        "State-of-the-art speaker recognition with neural network embeddings in 
        NIST SRE18 and Speakers in the Wild evaluations"
        https://www.sciencedirect.com/science/article/pii/S0885230819302700
        '''
        super(FTDNN, self).__init__()

        self.layer01 = TDNNLayer(input_dim=in_dim, output_dim=512, context_size=5, padding=2)
        self.layer02 = FTDNNLayer(512, 1024, 256, context_size=2, dilations=[2, 2, 2], paddings=[1, 1, 1])
        self.layer03 = FTDNNLayer(1024, 1024, 256, context_size=1, dilations=[1, 1, 1], paddings=[0, 0, 0])
        self.layer04 = FTDNNLayer(1024, 1024, 256, context_size=2, dilations=[3, 3, 2], paddings=[2, 1, 1])
        self.layer05 = FTDNNLayer(2048, 1024, 256, context_size=1, dilations=[1, 1, 1], paddings=[0, 0, 0])
        self.layer06 = FTDNNLayer(1024, 1024, 256, context_size=2, dilations=[3, 3, 2], paddings=[2, 1, 1])
        self.layer07 = FTDNNLayer(3072, 1024, 256, context_size=2, dilations=[3, 3, 2], paddings=[2, 1, 1])
        self.layer08 = FTDNNLayer(1024, 1024, 256, context_size=2, dilations=[3, 3, 2], paddings=[2, 1, 1])
        self.layer09 = FTDNNLayer(3072, 1024, 256, context_size=1, dilations=[1, 1, 1], paddings=[0, 0, 0])
        self.layer10 = DenseReLU(1024, 2048)

    def forward(self, x):
        '''
        Input must be (batch_size, seq_len, in_dim)
        '''
        x = self.layer01(x)
        x2 = self.layer02(x)
        x = self.layer03(x2)
        x4 = self.layer04(x)
        x = self.layer05(torch.cat([x4, x], dim=1))
        x6 = self.layer06(x)
        x = self.layer07(torch.cat([x6, x4, x2], dim=1))
        x8 = self.layer08(x)
        x = self.layer09(torch.cat([x8, x6, x4], dim=1))
        x = self.layer10(x)
        return x

    def step_ftdnn_layers(self):
        for layer in self.children():
            if isinstance(layer, FTDNNLayer):
                layer.step_semi_orth()

    def set_dropout_alpha(self, alpha):
        for layer in self.children():
            if isinstance(layer, FTDNNLayer):
                layer.dropout.alpha = alpha

    def get_orth_errors(self):
        errors = 0.
        with torch.no_grad():
            for layer in self.children():
                if isinstance(layer, FTDNNLayer):
                    errors += layer.orth_error()
        return errors
###################################
##### TDNN #######################
##################################

class TDNN(nn.Module):
    
    def __init__(
                    self, 
                    input_dim=23, 
                    output_dim=512,
                    context_size=5,
                    stride=1,
                    dilation=1,
                    batch_norm=True,
                    dropout_p=0.0,
                    padding=0
                ):
        super(TDNN, self).__init__()
        self.context_size = context_size
        self.stride = stride
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.dilation = dilation
        self.dropout_p = dropout_p
        self.padding = padding
        
        self.kernel = nn.Conv1d(self.input_dim, 
                                self.output_dim,
                                self.context_size, 
                                stride=self.stride, 
                                padding=self.padding, 
                                dilation=self.dilation)

        self.nonlinearity = nn.LeakyReLU()
        self.batch_norm = batch_norm
        if batch_norm:
            self.bn = nn.BatchNorm1d(output_dim)
        self.drop = nn.Dropout(p=self.dropout_p)
        
    def forward(self, x):
        '''
        input: size (batch, seq_len, input_features)
        outpu: size (batch, new_seq_len, output_features)
        '''

        _, _, d = x.shape
        assert (d == self.input_dim), 'Input dimension was wrong. Expected ({}), got ({})'.format(self.input_dim, d)
        
        x = self.kernel(x.transpose(1,2))
        x = self.nonlinearity(x)
        x = self.drop(x)

        if self.batch_norm:           
            x = self.bn(x)
        return x.transpose(1,2)

    
class ETDNN(nn.Module):

    def __init__(
                    self,
                    features_per_frame=80,
                    hidden_features=1024,
                    dropout_p=0.0,
                    batch_norm=True
                ):
        super(ETDNN, self).__init__()
        self.features_per_frame = features_per_frame
        self.hidden_features = hidden_features

        self.dropout_p = dropout_p
        self.batch_norm = batch_norm
        tdnn_kwargs = {'dropout_p':dropout_p, 'batch_norm':self.batch_norm}
        self.nl = nn.LeakyReLU()

        self.frame1 = TDNN(input_dim=self.features_per_frame, output_dim=self.hidden_features, context_size=5, dilation=1, **tdnn_kwargs)
        self.frame2 = TDNN(input_dim=self.hidden_features, output_dim=self.hidden_features, context_size=1, dilation=1, **tdnn_kwargs)
        self.frame3 = TDNN(input_dim=self.hidden_features, output_dim=self.hidden_features, context_size=3, dilation=2, **tdnn_kwargs)
        self.frame4 = TDNN(input_dim=self.hidden_features, output_dim=self.hidden_features, context_size=1, dilation=1, **tdnn_kwargs)
        self.frame5 = TDNN(input_dim=self.hidden_features, output_dim=self.hidden_features, context_size=3, dilation=3, **tdnn_kwargs)
        self.frame6 = TDNN(input_dim=self.hidden_features, output_dim=self.hidden_features, context_size=1, dilation=1, **tdnn_kwargs)
        self.frame7 = TDNN(input_dim=self.hidden_features, output_dim=self.hidden_features, context_size=3, dilation=4, **tdnn_kwargs)
        self.frame8 = TDNN(input_dim=self.hidden_features, output_dim=self.hidden_features, context_size=1, dilation=1, **tdnn_kwargs)
        self.frame9 = TDNN(input_dim=self.hidden_features, output_dim=self.hidden_features*3, context_size=1, dilation=1, **tdnn_kwargs)

        self.tdnn_list = nn.Sequential(self.frame1, self.frame2, self.frame3, self.frame4, self.frame5, self.frame6, self.frame7, self.frame8, self.frame9)

    def forward(self, x):
        x = self.tdnn_list(x)
        return x