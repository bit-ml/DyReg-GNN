import torch
from torch import nn
from torch.nn import functional as F

import matplotlib.pyplot as plt
import matplotlib.colors as colors

import sys
import os
sys.path.append(os.path.abspath(os.path.join(
    os.path.dirname(os.path.realpath(__file__)), '..')))
import time
from opts import parser

global args, best_prec1
from opts import parse_args
args = parser.parse_args()
args = parse_args()

import pickle
import pdb

if args.rnn_type == 'LSTM':
    recurrent_net = torch.nn.LSTM
elif args.rnn_type == 'GRU':
    recurrent_net = torch.nn.GRU

def differentiable_resize_area(x, kernel):
    # x: B x C x H x W
    # kernel: B x h x w x H x W
    B = x.size()[0]
    C = x.size()[1]
    H = x.size()[2]
    W = x.size()[3]
    h = kernel.size()[1]
    w = kernel.size()[2]

    kernel_res = kernel.view(B, h * w, H * W)
    kernel_res = kernel_res.permute(0,2,1)
    x_res = x.view(B, C, H * W)

    x_resize = torch.matmul(x_res, kernel_res)
    x_resize = x_resize.view(B, C, h , w)
    # x_resize: B x C x h x w
    return x_resize

def save_kernel(mean_kernels, folder='.'):
    kernel = mean_kernels[0].detach().numpy()
    max_t = kernel.shape[0]
    num_rows = kernel.shape[1]

    #for tt in range(max_t-5,max_t):
    all_frames = []
    for tt in range(max_t):
        f, axarr = plt.subplots(num_rows,num_rows)
        N = kernel.shape[1] * kernel.shape[2]
        for ii in range(kernel.shape[1]):
            for jj in range(kernel.shape[2]):
                axarr[ii][jj].imshow(kernel[tt][ii][jj])
    
        plt.savefig(f'{folder}/kernel_mean_{tt}.png')  


def atanh(x: torch.Tensor):
    x = x.clamp(-1 + 1e-7, 1 - 1e-7)
    return (torch.log(1 + x).sub(torch.log(1 - x))).mul(0.5)


class LayerNormAffine2D(nn.Module):
    # B * T x num_chan x H x W
    def __init__(self, num_ch, norm_shape, zero_init=False):
        super(LayerNormAffine2D, self).__init__()
        self.scale = torch.nn.Parameter(torch.Tensor(size=[1,num_ch,1,1]))#.to(input.device)
        self.bias = torch.nn.Parameter(torch.Tensor(size=[1,num_ch,1,1]))#.to(input.device)
        self.norm = nn.LayerNorm(norm_shape, elementwise_affine=False)#.to(input.device)
        
        bias_init = 0
        scale_init = 1
        if zero_init:
            scale_init = 0
        nn.init.constant_(self.bias, bias_init)
        nn.init.constant_(self.scale, scale_init)

    def forward(self, input):
        input = self.norm(input) * self.scale + self.bias
        return input

class LayerNormAffine1D(nn.Module):
    # B * T x num_chan x N
    def __init__(self, num_ch, norm_shape, zero_init=False):
        super(LayerNormAffine1D, self).__init__()
        self.scale = torch.nn.Parameter(torch.Tensor(size=[1,num_ch,1]))#.to(input.device)
        self.bias = torch.nn.Parameter(torch.Tensor(size=[1,num_ch,1]))#.to(input.device)
        self.norm = nn.LayerNorm(norm_shape, elementwise_affine=False)#.to(input.device)
        
        bias_init = 0
        scale_init = 1
        if zero_init:
            scale_init = 0
        nn.init.constant_(self.bias, bias_init)
        nn.init.constant_(self.scale, scale_init)

    def forward(self, input):
        input = self.norm(input) * self.scale + self.bias
        return input

# input: D1 x D2 x ... x C
# ex  B x N x T x C 
#     or B x N x N x T x C
class LayerNormAffineXC(nn.Module):
    def __init__(self, num_ch, norm_shape):
        super(LayerNormAffineXC, self).__init__()
        self.scale = torch.nn.Parameter(torch.Tensor(size=[num_ch]))
        self.bias = torch.nn.Parameter(torch.Tensor(size=[num_ch]))
        self.norm = nn.LayerNorm(norm_shape, elementwise_affine=False)
    
        nn.init.constant_(self.bias, 0)
        nn.init.constant_(self.scale, 1)

    def forward(self, input):
        input = self.norm(input) * self.scale + self.bias
        return input




class GloRe(nn.Module):
    def __init__(self, input_height=14, input_channels=16, h=3,w=3):
        super(GloRe, self).__init__()

        self.h = h
        self.w = w
        self.H = input_height
        self.W = input_height
        self.input_channels     = input_channels
        self.num_nodes          = h * w
        
        self.norm_dict = nn.ModuleDict({})
        self.norm_dict[f'offset_location_emb'] = LayerNormAffine2D(
            input_channels, (input_channels, self.h, self.w)
        )
        # reduce channels
        self.mask_conv      = nn.Conv2d(self.input_channels, self.num_nodes, [1, 1]) 
        self.update         = nn.Conv1d(self.input_channels, args.offset_lstm_dim, [1])

        # positional embeding
        pos_emb = self.positional_emb(self.input_channels)
        self.register_buffer('pos_emb_buf', pos_emb)
        self.sin_pos_emb_proj = nn.Conv2d(self.input_channels, self.input_channels, [1,1])

    def get_norm(self, input, name):
        # input: B * T x C x H x W
        norm = self.norm_dict[name]
        input = norm(input)
        return input
    def apply_sin_positional_emb(self, x):
        # pos_emb_buf: C x H x W
        # x:B * T x C x H x W   
        emb = self.pos_emb_buf.unsqueeze(0)
        
        emb = emb.repeat(x.shape[0],1,1,1)
        emb = self.sin_pos_emb_proj(emb)
        out = x + emb
        return out  #emb- just pos
    def positional_emb(self, num_channels=2*32):
        # pos_h:        H x 1 x 1
        # T:          1 x C x 1
        # T_even:     1 x C/2 x 1
        # emb_h_even:   H x C/2 x 1
        channels = num_channels // 2
        pos_h = torch.linspace(0,1,self.H).unsqueeze(1).unsqueeze(2)
        T = torch.pow(10000, torch.arange(0,channels).float() / channels)
        T_even = T.view(-1,2)[:,:1].unsqueeze(0)
        emb_h_even = torch.sin(pos_h / T_even)

        T_odd = T.view(-1,2)[:,1:].unsqueeze(0)
        emb_h_odd = torch.cos(pos_h / T_odd)
    
        emb_h = torch.cat((emb_h_even, emb_h_odd), dim=2).view(self.H,1, channels)
        emb_h = emb_h.repeat(1,self.W,1)

        # pos_w:        W x 1 x 1
        # T:          1 x C x 1
        # T_even:     1 x C/2 x 1
        # emb_h_even:   W x C/2 x 1
        pos_w = torch.linspace(0,1,self.W).unsqueeze(1).unsqueeze(2)
        emb_w_even = torch.sin(pos_w / T_even)
        emb_w_odd  = torch.cos(pos_w / T_odd)
        emb_w = torch.cat((emb_w_even, emb_w_odd), dim=2).view(1, self.W, channels)
        emb_w = emb_w.repeat(self.H,1,1)

        emb = torch.cat((emb_h, emb_w), dim=2).permute(2,0,1)
        return emb
    def forward(self, x):
        # x: B * T x C x H x W 
        # mask: B * T x num_nodes x H x W
        mask = self.mask_conv(x) 
        x = self.apply_sin_positional_emb(x)

        # softmax over the number of receiving nodes: each of the 
        # H * W location send predominantly to one node
        mask = F.softmax(mask, dim=1)
        mask = mask.view(mask.shape[0], 3,3, mask.shape[2], mask.shape[3])

        # nodes: B*T x C x num_nodes
        offset_nodes = differentiable_resize_area(x, mask)

        # B*T x C x h x w
        offset_nodes = offset_nodes.view(offset_nodes.shape[0], offset_nodes.shape[1], offset_nodes.shape[2] * offset_nodes.shape[3])
        offset_nodes = self.update(offset_nodes)
        # nodes: B*T x offset_lstm_dim x num_nodes
        return offset_nodes


def get_fishnet_params(input_height,keep_spatial_size = False):
    first_height             = input_height
    if input_height == 32:
        first_height = 32
        strides = [(2, 2), (2,2)]
        padding = [(0,0), (0,0)]
        norm_size = [15, 7, 7]
    elif input_height == 16:
        first_height = 16
        strides = [(2, 2), (1,1)]
        padding = [(0,0), (1,1)]
        norm_size = [7, 7, 7]
    elif input_height == 8:
        strides = [(1, 1), (1,1)]
        padding = [(0,0), (1,1)]
        norm_size = [6, 6, 6]
    elif input_height > 50:
        strides = [(2, 2), (2,2)]
        padding = [(0,0), (0,0)]
        norm_size = [13, 6, 6]
        self.max_pool = torch.nn.MaxPool2d(2, stride=2)
        first_height = 28
    elif input_height > 20:
        strides         = [(2, 2), (2,2)]
        padding = [(0,0), (0,0)]
        norm_size = [13, 6, 6]
    elif input_height > 10:
        strides         = [(2, 2), (1,1)]
        padding = [(0,0), (1,1)]
        norm_size = [6, 6, 6]
    else:
        strides         = [(1, 1), (1,1)]
        padding = [(0,0), (1,1)]
        norm_size = [5, 5, 5]

    output_padding = (0,0)
    if keep_spatial_size:
        if input_height == 16:
            padding = [(1,1), (1,1)]
            output_padding = (1,1)
            norm_size = [8, 8, 16] 
    return first_height, strides, padding, norm_size, output_padding
    
class Fishnet(nn.Module):
    def __init__(self, input_height=14, input_channels=16, keep_spatial_size=False):
        super(Fishnet, self).__init__()

        self.offset_layers       = 2
        self.offset_channels     = [32, 16, 16]
        self.offset_channels_tr  = [32, 32, 16]
        self.input_height        = input_height
        self.keep_spatial_size   = keep_spatial_size
        
        first_height, strides, padding, norm_size, output_padding = get_fishnet_params(
            input_height, keep_spatial_size)
        self.norm_size = norm_size

        self.norm_dict = nn.ModuleDict({})

        # reduce channels
        self.conv1 = nn.Conv2d(input_channels, self.offset_channels[0], [1, 1]) # de pus relu
        self.norm_dict[f'norm1'] = LayerNormAffine2D(
            self.offset_channels[0], 
            (self.offset_channels[0], first_height, first_height)
        )

        self.tail_conv = nn.ModuleList() 
        self.body_trans_conv = nn.ModuleList() 
        self.head_conv = nn.ModuleList()
        for i in range(self.offset_layers):
            self.tail_conv.append(nn.Conv2d(
                self.offset_channels[max(0,i-1)], self.offset_channels[i],
                [3, 3], padding=padding[i], stride=strides[i]
                )
            )
            self.norm_dict[f'tail_norm_{i}'] = LayerNormAffine2D(
                self.offset_channels[i], 
                (self.offset_channels[i], norm_size[i], norm_size[i])
            )
        stop = 0
        if self.keep_spatial_size:
            stop = -1
        for ind, i in enumerate(range(self.offset_layers - 1, stop,-1)):    
                if keep_spatial_size and i == 0:
                    self.body_trans_conv.append(nn.ConvTranspose2d(
                        self.offset_channels_tr[i+1], self.offset_channels_tr[i],
                        [3, 3], padding=padding[i], output_padding=output_padding,
                        stride=strides[i]
                        )
                    )
                else:
                    self.body_trans_conv.append(nn.ConvTranspose2d(
                        self.offset_channels_tr[i+1], self.offset_channels_tr[i],
                        [3, 3], padding=padding[i], stride=strides[i]
                        )
                    )

                self.norm_dict[f'body_trans_norm_{ind}'] = LayerNormAffine2D(
                    self.offset_channels_tr[i],
                    (self.offset_channels_tr[i], norm_size[i-1], norm_size[i-1])
                )

        if not self.keep_spatial_size:
            for i in range(1, self.offset_layers):
                self.head_conv.append(nn.Conv2d(
                    self.offset_channels[i-1], self.offset_channels[i],
                    [3, 3], padding=padding[i], stride=strides[i]
                    )
                )
                self.norm_dict[f'head_norm_{i-1}'] = LayerNormAffine2D(
                    self.offset_channels[i], 
                    (self.offset_channels[i], norm_size[i], norm_size[i])
                )

    def get_norm(self, input, name, zero_init=False):
        # input: B * T x C x H x W
        norm = self.norm_dict[name]
        input = norm(input)
        return input

    def forward(self, x):
        # x: B * T x C x H x W 
        x = F.relu(self.conv1(x))
        if self.input_height > 50:
            x = self.max_pool(x)

        x = self.get_norm(x, 'norm1')
        all_x = []
        all_x = [x]
        
        for i in range(self.offset_layers):
            x = F.relu(self.tail_conv[i](x))
            x = self.get_norm(x, f'tail_norm_{i}')
            all_x.append(x)

        stop = 0
        if self.keep_spatial_size:
            stop = -1

        for ind, i in enumerate(range(self.offset_layers - 1, stop,-1)): 
            x = F.relu(self.body_trans_conv[ind](x))
            x = self.get_norm(x, f'body_trans_norm_{ind}')
            x = x + all_x[i]

        if not self.keep_spatial_size:
            for i in range(1, self.offset_layers):
                x = F.relu(self.head_conv[i-1](x))
                x = self.get_norm(x, f'head_norm_{i-1}')

        return x