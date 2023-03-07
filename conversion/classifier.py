import torch
from torch.nn import functional as F
from torch.nn import Dropout, Sequential, Linear, Softmax
import hparams as hp
import torch.nn as nn

class GradientReversalFunction(torch.autograd.Function):
    """Revert gradient without any further input modification."""

    @staticmethod
    def forward(ctx, x, l, c):
        ctx.l = l
        ctx.c = c
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        grad_output = grad_output.clamp(-ctx.c, ctx.c)
        return ctx.l * grad_output.neg(), None, None


class GradientClippingFunction(torch.autograd.Function):
    """Clip gradient without any further input modification."""

    @staticmethod
    def forward(ctx, x, c):
        ctx.c = c
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        grad_output = grad_output.clamp(-ctx.c, ctx.c)
        return grad_output, None


# class ReversalClassifier(torch.nn.Module):
#     """Adversarial classifier (with two FC layers) with a gradient reversal layer.
    
#     Arguments:
#         input_dim -- size of the input layer (probably should match the output size of encoder)
#         hidden_dim -- size of the hiden layer
#         output_dim -- number of channels of the output (probably should match the number of speakers/languages)
#         gradient_clipping_bound (float) -- maximal value of the gradient which flows from this module
#     Keyword arguments:
#         scale_factor (float, default: 1.0)-- scale multiplier of the reversed gradientts
#     """

#     def __init__(self, input_dim=256, hidden_dim=256, output_dim=173, gradient_clipping_bounds=0.25, scale_factor=1.0):
#         super(ReversalClassifier, self).__init__()
#         self._lambda = scale_factor
#         self._clipping = gradient_clipping_bounds
#         self._output_dim = output_dim
#         self._classifier = Sequential(
#             Linear(input_dim, hidden_dim),
#             Linear(hidden_dim, output_dim)
#         )
#         #self.loss = nn.CrossEntropyLoss(reduction='none')
#     def forward(self, x):  
#         x = GradientReversalFunction.apply(x, self._lambda, self._clipping)
#         x = self._classifier(x)
#         return x

#     @staticmethod
#     def loss(input_lengths1, input_lengths2, speakers1, speakers2, prediction1, prediction2, embeddings=None):
#         ignore_index = -100
#         ml = torch.max(input_lengths)
#         input_mask = torch.arange(ml, device=input_lengths.device)[None, :] < input_lengths[:, None]
#         #print(input_mask.size())

#         #print(prediction.size())
#         #target = speakers.unsqueeze(1).expand(-1, ml).reshape(-1)
#         target = speakers.repeat(1, ml)#.transpose(0,1)
#         #print(target.size())
#         target[~input_mask] = ignore_index
#         return F.cross_entropy(prediction.transpose(1,2), target, ignore_index=ignore_index)
#         #return self.

class ReversalClassifier(torch.nn.Module):
    """Adversarial classifier (with two FC layers) with a gradient reversal layer.
    
    Arguments:
        input_dim -- size of the input layer (probably should match the output size of encoder)
        hidden_dim -- size of the hiden layer
        output_dim -- number of channels of the output (probably should match the number of speakers/languages)
        gradient_clipping_bound (float) -- maximal value of the gradient which flows from this module
    Keyword arguments:
        scale_factor (float, default: 1.0)-- scale multiplier of the reversed gradientts
    """

    def __init__(self, input_dim=256, hidden_dim=256, output_dim=hp.num_speaker, gradient_clipping_bounds=0.25, scale_factor=1.0):
        super(ReversalClassifier, self).__init__()
        self._lambda = scale_factor
        self._clipping = gradient_clipping_bounds
        self._output_dim = output_dim
        self._classifier = Sequential(
            Linear(input_dim, hidden_dim),
            Linear(hidden_dim, output_dim)
        )
        #self.loss = nn.CrossEntropyLoss(reduction='none')
    def forward(self, x):  
        x = GradientReversalFunction.apply(x, self._lambda, self._clipping)
        x = self._classifier(x)
        return x

    @staticmethod
    def loss(input_lengths, speakers, prediction, embeddings=None):
        ignore_index = -100
        ml = torch.max(input_lengths)
        input_mask = torch.arange(ml, device=input_lengths.device)[None, :] < input_lengths[:, None]

#         target = speakers.unsqueeze(1).expand(-1, ml).reshape(-1)
        target = speakers.unsqueeze(1).repeat(1, ml)#.transpose(0,1)
#         print(target.size())
        target[~input_mask] = ignore_index
#         print(prediction.size(),target.size(),'&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&')
        return F.cross_entropy(prediction.transpose(1,2), target, ignore_index=ignore_index)



class CosineSimilarityClassifier(torch.nn.Module):
    """Cosine similarity-based adversarial classifier.
    
    Arguments:
        input_dim -- size of the input layer (probably should match the output size of encoder)
        output_dim -- number of channels of the output (probably should match the number of speakers/languages)
        gradient_clipping_bound (float) -- maximal value of the gradient which flows from this module
    """

    def __init__(self, input_dim, output_dim, gradient_clipping_bounds):
        super(CosineSimilarityClassifier, self).__init__()
        self._classifier = Linear(input_dim, output_dim)
        self._clipping = gradient_clipping_bounds

    def forward(self, x):
        x = GradientClippingFunction.apply(x, self._clipping)
        return self._classifier(x)

    @staticmethod
    def loss(input_lengths, speakers, prediction, embeddings, instance):
        l = ReversalClassifier.loss(input_lengths, speakers, prediction)

        w = instance._classifier.weight.T # output x input

        dot = embeddings @ w
        norm_e = torch.norm(embeddings, 2, 2).unsqueeze(-1)
        cosine_loss = torch.div(dot, norm_e)
        norm_w = torch.norm(w, 2, 0).view(1, 1, -1)
        cosine_loss = torch.div(cosine_loss, norm_w)
        cosine_loss = torch.abs(cosine_loss)

        cosine_loss = torch.sum(cosine_loss, dim=2)
        l += torch.mean(cosine_loss)
        
        return l
