import math, copy, time
from torch.autograd import Variable

import torch
import torch.nn as nn

import torch.nn.functional as F
import numpy as np


class Wrapper(nn.Module):
    def __init__(self, weight_norm=True):
        super(Wrapper, self).__init__()
        self._name = 'Wrapper'
        self.weight_norm = weight_norm

    def forward(self, x):
        if self.weight_norm:
            return nn.utils.weight_norm(x)
        else:
            return x

# def clones(module, N):
#     "Produce N identical layers."
#     return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


def weights_init_selu(model):

    if isinstance(model, nn.Linear):
        print('init weights selu')
        nb_inputs = float(model.weight.data.size(0))
        s = np.sqrt(nb_inputs)
        torch.nn.init.normal_(model.weight.data, mean=0.0, std= 1.0 / s)
        #torch.nn.init.uniform_(model.weight.data, a=-1.0 / s, b=1.0 / s)
        model.weight.data -= torch.mean(model.weight.data) #, dim=0, keepdims=True)
        model.weight.data /= s*torch.std(model.weight.data)

        if not model.bias is None:
            torch.nn.init.zeros_(model.bias.data)

def weights_init_glorot(tensor):
    fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(tensor)
    std = math.sqrt(2.0 / float(fan_in + fan_out))
    a = math.sqrt(3.0) * std  # Calculate uniform bounds from standard deviation
    w = nn.init._no_grad_uniform_(tensor, -a, a)
    return w - torch.mean(w)


def attention(query, key, value, mask=None, dropout=None):
    "Compute 'Scaled Dot Product Attention'"
    d_k = query.size(-1)

    scores = torch.matmul(query, key.transpose(-2, -1))/ math.sqrt(d_k)

    if mask is not None:
        #print(scores.size(), mask.size())
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim = -1)
    #p_attn = torch.tanh(scores) #/ query.size(-2)
    if dropout is not None:
        p_attn = dropout(p_attn)

    #print(p_attn.size(), value.size())
    x = torch.matmul(p_attn, value)
    #print(query.size(), key.transpose(-2, -1).size(), scores.size(), x.size())
    return x, p_attn


class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.b = 2

        self.qkv = nn.Linear(d_model, d_model * 3 * self.b, bias=False)
        nn.init.xavier_uniform_(self.qkv.weight)

        # self.linears0 = nn.Linear(d_model, d_model)
        # self.linears1 = nn.Linear(d_model, d_model)
        # self.linears2 = nn.Linear(d_model, d_model, bias=False)
        #self.linears3 = nn.Linear(d_model, d_model)
        # self.linears0 = Wrapper()(nn.Linear(d_model, d_model))
        # self.linears1 = Wrapper()(nn.Linear(d_model, d_model))
        #self.linears2 = nn.Linear(d_model, self.b*d_model, bias=False)
        self.linears3 = Wrapper()(nn.Linear(self.b*d_model, 2*d_model, bias=False))
        nn.init.xavier_normal_(self.linears3.weight)
        self.linears4 = Wrapper()(nn.Linear(2*d_model, d_model, bias=False))
        nn.init.xavier_normal_(self.linears4.weight)
        #self.linears4 = Wrapper()(nn.Linear(self.b*d_model, d_model, bias=False))

        #self.linears = clones(nn.Linear(d_model, d_model, bias=False), 4)
        self.attn = None
        #self.dropout = nn.Dropout(p=dropout)

        self.dropout = nn.Dropout(dropout)


        #nn.init.zeros_(self.qkv.bias)



        # for p in self.parameters():
        #    if p.dim() > 1:
        #        nn.init.xavier_uniform_(p)
        #    else:
        #        nn.init.zeros_(p)


    def forward(self, x, mask=None):
        "Implements Figure 2"
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = x.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        # query, key, value = \
        #     [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
        #      for l, x in zip((self.linears0, self.linears1, self.linears2), (x, x, x))]
        B, N, C = x.shape
        query, key, value = self.qkv(x).reshape(B, N, 3, self.b *self.h, C // self.h).permute(2, 0, 3, 1, 4)
        #query, key = self.qkv(x).reshape(B, N, 2, self.b * self.h, C // self.h).permute(2, 0, 3, 1, 4)
        #value = self.linears2(x).reshape(B, N, self.b*self.h, C // self.h).permute(0, 2, 1, 3)

        # 2) Apply attention on all the projected vectors in batch.
        x, self.attn = attention(query, key, value, mask=mask,
                                 dropout=self.dropout)

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous() \
            .view(nbatches, -1, self.b*self.h * self.d_k)

        #return self.linears3(x)
        return self.linears4(F.gelu(self.linears3(x)))
        #return self.linears4(F.selu(self.linears3(x)))
        #return self.linears4(F.selu(self.linears3(x)))
        #return self.linears4(x)


class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()
        self._name = 'Flatten'

    def forward(self, inputs):
        return inputs.reshape(inputs.size(0), -1)

class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = Wrapper()(nn.Linear(d_model, d_ff, bias=False))
        #self.w_1b = Wrapper()(nn.Linear(d_ff, d_ff, bias=False))
        self.w_2 = Wrapper()(nn.Linear(d_ff, d_model, bias=False))
        #self.w_1 = nn.Linear(d_model, d_ff, bias=False)
        #self.w_2 = nn.Linear(d_ff, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)
        #self.lnorm1 = nn.LayerNorm((404, d_model))
        #self.lnorm2 = nn.LayerNorm((404, d_model))

        #self.w_1.apply(weights_init_selu)
        #self.w_2.apply(weights_init_selu)

        for p in self.parameters():
           if p.dim() > 1:
               nn.init.xavier_uniform_(p)
           else:
               nn.init.zeros_(p)

    def forward(self, x):
        # return self.lnorm2(self.w_2(self.dropout(F.selu(self.lnorm1(self.w_1(x))))))
        return self.w_2(self.dropout(F.gelu(self.w_1(x))))
        #return self.w_2(F.selu(self.w_1b(self.dropout(F.selu(self.w_1(x))))))






class Identity(nn.Module):
    "Implements FFN equation."
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x







class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = Identity() #LayerNorm(size) #
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        #return self.dropout(sublayer(self.norm(x)))
        return x + self.dropout(sublayer(self.norm(x)))


class EncoderLayer(nn.Module):
    "Encoder is made up of self-attn and feed forward (defined below)"
    def __init__(self, d_model, heads, d_ff, dropout=0.05):
        super(EncoderLayer, self).__init__()
        self.self_attn = MultiHeadedAttention(heads, d_model)
        #self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        #self.self_attn = self_attn
        #self.feed_forward = feed_forward
        #self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.sublayer0 = SublayerConnection(d_model, dropout)
        #self.sublayer1 = SublayerConnection(d_model, dropout)


    def forward(self, x, mask):
        "Follow Figure 1 (left) for connections."
        x = self.sublayer0(x, lambda x: self.self_attn(x, mask))
        return x
        #return self.sublayer1(x, self.feed_forward)



class Encoder(nn.Module):
    "Core encoder is a stack of N layers"

    def __init__(self, d_model, heads, d_ff, dropout, N):
        super(Encoder, self).__init__()
        #self.layers = clones(layer, N)
        #self.layers = [EncoderLayer(d_model, heads, d_ff, dropout) for i in range(N)]
        #self.norm = LayerNorm(d_model) #Identity() #

        self.layers = {}
        for j in range(0, N):
            #self.layers[j] = EncoderLayer(d_model, heads, d_ff, dropout)
            self.layers[j] = EncoderLayer(d_model, heads[j], d_ff, dropout)
            self.add_module('layer_' + str(j), self.layers[j])

    def forward(self, x, mask):
        "Pass the input (and mask) through each layer in turn."
        for j, layer in self.layers.items():
            x = layer(x, mask)
        #return self.norm(x)
        return x


class LastLayer(nn.Module):
    "Map from the embedding space back to the input space"

    def __init__(self, d_model, n_channels, activation='sigmoid'):
        super(LastLayer, self).__init__()
        self.d_model = d_model
        self.n_channels = n_channels
        self.activation = activation
        self.proj = Wrapper()(nn.Linear(d_model, n_channels, bias=False))

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
                # nn.init.xavier_normal_(p)
            else:
                nn.init.zeros_(p)

    def forward(self, x, mask):
        #(N, L, _) = x.size()
        #x = x.reshape(N*L, self.d_model)
        x = self.proj(x)
        #x = self.proj(x)
        #x = x.reshape(N, L, self.n_channels)
        if self.activation == 'double_gelu':
            x = F.gelu(x)
            x = 1 - F.gelu(1-x)
        else:
            x = torch.sigmoid(x)
        return x


class PosEmb(nn.Module):
    "construct pixel position embeddings"
    def __init__(self, d_model, size=(20, 20), mode='standard', cat=True):
        super(PosEmb, self).__init__()
        self.d_model = d_model
        self.size = size
        self.mode = mode
        self.cat = cat

        if self.mode == 'pairwise' :
            self.posencx = nn.utils.weight_norm(nn.Embedding(size[0]//2, d_model))
            self.posency = nn.utils.weight_norm(nn.Embedding(size[1]//2, d_model))
            self.posencx2 = nn.utils.weight_norm(nn.Embedding(size[0]//2+1, d_model))
            self.posency2 = nn.utils.weight_norm(nn.Embedding(size[1]//2+1, d_model))
        elif self.mode == 'sequential':
            self.posenc = nn.utils.weight_norm(nn.Embedding(size[0]*size[1], d_model))
        else:
            self.posencx = nn.utils.weight_norm(nn.Embedding(size[0], d_model))
            self.posency = nn.utils.weight_norm(nn.Embedding(size[1], d_model))


    def forward(self, inp):
        B = inp.size(0)
        H = self.size[0]
        W = self.size[1]
        if inp.is_cuda:
            cuda = 'cuda'
        else:
            cuda = None

        if self.mode == 'pairwise':
            xemb = self.posencx(torch.arange(H//2, device=cuda).reshape(H//2, 1).repeat(1, 2).reshape(-1))
            xpos = xemb.unsqueeze(0).repeat(W, 1, 1)
            xpos = xpos.reshape(W*H, -1)
            xpos = xpos.unsqueeze(0).repeat(B, 1, 1)

            xemb2 = self.posencx2(torch.arange(H//2+1, device=cuda).reshape(H//2+1, 1).repeat(1, 2).reshape(-1)[1:H+1])
            xpos2 = xemb2.unsqueeze(0).repeat(W, 1, 1)
            xpos2 = xpos2.reshape(W*H, -1)
            xpos2 = xpos2.unsqueeze(0).repeat(B, 1, 1)

            yemb = self.posency(torch.arange(W // 2, device=cuda).reshape(W // 2, 1).repeat(1, 2).reshape(-1))
            ypos = yemb.unsqueeze(1).repeat(1, H, 1)
            ypos = ypos.reshape(W*H, -1)
            ypos = ypos.unsqueeze(0).repeat(B, 1, 1)

            yemb2 = self.posency2(torch.arange(W // 2 + 1, device=cuda).reshape(W // 2 + 1, 1).repeat(1, 2).reshape(-1)[1:W+1])
            ypos2 = yemb2.unsqueeze(1).repeat(1, H, 1)
            ypos2 = ypos2.reshape(W*H, -1)
            ypos2 = ypos2.unsqueeze(0).repeat(B, 1, 1)

            if self.cat:
                x = torch.cat([xpos, ypos, xpos2, ypos2], dim=2)
            else:
                x = xpos + ypos + xpos2 + ypos2

        elif self.mode == 'sequential':
            x = self.posenc(torch.arange(H*W, device=cuda))
            x = x.unsqueeze(0).repeat(B, 1, 1)

        else:
            xemb = self.posencx(torch.arange(H, device=cuda))
            xpos = xemb.unsqueeze(0).repeat(W, 1, 1)
            xpos = xpos.reshape(W*H, -1)
            xpos = xpos.unsqueeze(0).repeat(B, 1, 1)

            yemb = self.posency(torch.arange(W, device=cuda))
            ypos = yemb.unsqueeze(1).repeat(1, H, 1)
            ypos = ypos.reshape(W*H, -1)
            ypos = ypos.unsqueeze(0).repeat(B, 1, 1)

            if self.cat:
                x = torch.cat([xpos, ypos], dim=2)
            else:
                x = xpos+ypos

        return x


class PixelEmb(nn.Module):
    "construct pixel position embeddings"
    def __init__(self, d_model, size=(20, 20), mode='standard', cat=True):
        super(PixelEmb, self).__init__()
        self.d_model = d_model
        self.size = size
        self.cat = cat
        self.posem = PosEmb(self.d_model, self.size, mode=mode, cat=cat)

        self.map = nn.utils.weight_norm(nn.Linear(1, d_model, bias=True))

        if self.cat:
            f = 3
            if mode == 'sequential':
                f = 2
            elif mode == 'pairwise':
                f = 5
            self.linear = nn.utils.weight_norm(nn.Linear(f*d_model, d_model, bias=False))

    def forward(self, inp):
        pix = self.map(inp.unsqueeze(-1))
        pos = self.posem(inp)
        if self.cat:
            x = torch.cat([pix, pos], dim=2)
            x = self.linear(x)
        else:
            x = pix + pos
        return x


class VocabEmb(nn.Module):
    "construct pixel position embeddings"
    def __init__(self, d_model, size=(20, 20), mode='standard', cat=True):
        super(VocabEmb, self).__init__()
        self.d_model = d_model
        self.size = size
        self.cat = cat
        self.posem = PosEmb(self.d_model, self.size, mode=mode)

        if self.cat:
            f = 3
            if mode == 'sequential':
                f = 2
            elif mode == 'pairwise':
                f = 5

            self.mapper = nn.utils.weight_norm(nn.Linear(f*d_model, d_model, bias=False))

        self.map = nn.utils.weight_norm(nn.Embedding(2, d_model))

    def forward(self, inp):
        #pixel value embeddings
        pix = self.map(inp.long())

        #position embeddings
        pos = self.posem(inp)

        if self.cat:
            x = torch.cat([pix, pos], dim=2)
            x = self.mapper(x)
        else:
            x = pix + pos

        return x







class UnetMem(nn.Module):
    def __init__(self, n_layers=3, d_model=64, d_ff=64, h=[4, 4, 4], size=(20, 20), dropout=0.1, output_channels=1, mode='standard'):
        super(UnetMem, self).__init__()

        self.d_model = d_model
        self.flatten = Flatten()

        self.em = VocabEmb(d_model, size=size, mode=mode)
        #self.mememb = nn.utils.weight_norm(nn.Embedding(1, d_model)) #


        self.enc = Encoder(d_model, h, d_ff, dropout, n_layers)

        self.output = LastLayer(d_model, output_channels)


    def forward(self, input, mask):
        (B, C, W, H) = input.size()

        input = self.flatten(input)

        if not mask is None:
            mask = self.flatten(mask)
            input = input*mask

        x = self.em(input)

        #mem = self.mememb(torch.arange(1).cuda()).unsqueeze(0).repeat(B, 1, 1)
        #x = torch.cat([x, mem], dim=1)
        #x = torch.cat([x, torch.zeros_like(x[:, 0:1])], dim=1)

        # x = torch.cat([x, torch.zeros_like(x[:, 0:1])], dim=1)
        # if not mask is None:
        #     mask = torch.cat([mask, torch.ones_like(mask[:, 0:1])], dim=1)

        x = self.enc(x, mask)
        xt = self.output(x, mask)
        x = xt[:, 0:W*H]
        latent = None #xt[:, W*H:]
        return x.reshape((B, C, W, H)), latent



class AEncoder(nn.Module):
    def __init__(self, n_layers=3, d_model=64, d_ff=64, h=[4, 4, 4], size=(20, 20), dropout=0.1, output_channels=1):
        super(AEncoder, self).__init__()

        self.d_model = d_model
        self.flatten = Flatten()

        self.em = PixelEmb(d_model, size=size, mode='pairwise')

        self.enc = Encoder(d_model, h, d_ff, dropout, n_layers)
        #self.output = Wrapper()(nn.Linear(d_model, d_model, bias=False))


    def forward(self, input, mask=None):
        (B, C, W, H) = input.size()

        input = self.flatten(input)

        if not mask is None:
            mask = self.flatten(mask)
            input = input*mask

        x = self.em(input)

        x = torch.cat([x, torch.zeros_like(x[:, 0:1])], dim=1)
        if not mask is None:
            mask = torch.cat([mask, torch.ones_like(mask[:, 0:1])], dim=1)

        xt = self.enc(x, mask)

        latent_emb = xt[:, W*H:].reshape(B, -1)

        #latent_emb = self.output(F.selu(latent_emb))

        return latent_emb


class ADecoder(nn.Module):
    def __init__(self, n_layers=3, d_model=64, d_ff=64, h=[4, 4, 4], size=(20, 20), dropout=0.1, output_channels=1):
        super(ADecoder, self).__init__()
        self.C = 1
        self.W = size[0]
        self.H = size[1]
        self.d_model = d_model
        self.flatten = Flatten()

        self.em = PosEmb(d_model, size=size)
        self.mapper = nn.utils.weight_norm(nn.Linear(3*d_model, d_model, bias=False))

        self.enc = Encoder(d_model, h, d_ff, dropout, n_layers)
        self.output = LastLayer(d_model, output_channels, activation='double_gelu')



    def forward(self, input):
        B = input.size(0)
        posem = self.em(input)
        input = input.unsqueeze(1).repeat(1, self.W*self.H, 1)

        #print(posem.size(), input.size())
        x = torch.cat([input, posem], dim=2)
        x = self.mapper(x)

        x = self.enc(x, None)
        x = self.output(x, None)

        return x.reshape((B, self.C, self.W, self.H))



class Autoencoder(nn.Module):
    def __init__(self, n_layers=3, d_model=64, d_ff=64, h=[8, 8, 8], size=(20, 20), dropout=0.1, output_channels=1):
        super(Autoencoder, self).__init__()
        self.enc = AEncoder(n_layers=n_layers, d_model=d_model, h=h, size=size, dropout=dropout)
        self.dec = ADecoder(n_layers=n_layers, d_model=d_model, h=h, size=size, dropout=dropout)

    def forward(self, input):
        x = self.enc(input)
        x = self.dec(x)
        return x



class EDE(nn.Module):
    def __init__(self, n_layers=3, d_model=64, d_ff=64, h=[8, 8, 8], size=(20, 20), dropout=0.1, output_channels=1):
        super(EDE, self).__init__()
        self.enc = AEncoder(n_layers=n_layers, d_model=d_model, h=h, size=size, dropout=dropout)
        self.dec = ADecoder(n_layers=n_layers, d_model=d_model, h=h, size=size, dropout=0.0)

    def forward(self, input):
        mu = self.enc(input)
        x = self.dec(mu)
        return x


class Classifier(nn.Module):
    def __init__(self, n_layers=4, d_model=64, d_ff=64, h=[4, 4, 4, 4], size=(20, 20), dropout=0.1, n_outputs=10):
        super(Classifier, self).__init__()
        self.enc = AEncoder(n_layers=n_layers, d_model=d_model, d_ff=d_ff, h=h, size=size, dropout=dropout)
        self.output = nn.Sequential(
            nn.Linear(d_model, n_outputs, bias=True),
            )

    def forward(self, input):
        x = self.enc(input)
        x = self.output(x)
        #print(x.size())
        x = F.softmax(x, dim=-1)
        return x



