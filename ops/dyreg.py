# Code for Discovering Dynamic Salient Regions with Spatio-Temporal Graph Neural Networks
# https://arxiv.org/abs/2009.08427
# DyReG

import torch
from torch import nn
from torch.nn import functional as F

import matplotlib.pyplot as plt
import matplotlib.colors as colors

import sys
import os
sys.path.append(os.path.abspath(os.path.join(
    os.path.dirname(os.path.realpath(__file__)), '..')))
from opts import parser
from ops.rstg import *
from ops.models_utils import *

global args, best_prec1
from opts import parse_args
args = parser.parse_args()
args = parse_args()


class DynamicGraph(nn.Module):
    def __init__(self, 
            params,
            backbone_dim=1024, 
            H=14, W=14,oH=7,oW=7,iH=None,iW=None,
            h=3,w=3,
            node_dim=512, out_num_ch = 2048, project_i3d=False, name=''):
        super(DynamicGraph, self).__init__()
        self.params = params
        # the dimenstion of the input used at training time
        # to be used for resize for fullsize evaluation
        self.iH = iH
        self.iW = iW
        self.name = name
        self.rstg = RSTG(params, backbone_dim=backbone_dim, node_dim=node_dim, project_i3d=project_i3d)

        self.backbone_dim = backbone_dim
        self.C = self.rstg.node_dim
        self.out_num_ch = out_num_ch
        self.H = H
        self.W = W
      
        self.h = h
        self.w = w
        self.num_nodes = self.h * self.w
        self.project_i3d = project_i3d
        ph, pw = self.compute_anchors()
        self.register_buffer('ph_buf', ph)
        self.register_buffer('pw_buf', pw)

        self.norm_dict = nn.ModuleDict({})
        # input projection           
        if self.project_i3d:
            self.project_i3d_linear = nn.Conv2d(backbone_dim, self.C, [1, 1])
            self.project_back_i3d_linear = nn.Conv2d(self.C, backbone_dim, [1,1])

        self.create_offset_modules()

        # project kernel
        if self.params.full_res:
            self.kernel_projection1 = torch.nn.Linear(self.iH * self.iW, self.C)
            self.kernel_location_pool = nn.AdaptiveAvgPool2d((self.iH,self.iW))
        else:
            self.kernel_projection1 = torch.nn.Linear(self.H * self.W, self.C) 
        
        # B*T x 1 x H 
        arange_h = (torch.arange(0, self.H)
            .unsqueeze(0).unsqueeze(0)
            .to(self.kernel_projection1.weight.device)
        )
        # # B*T x 1 x W 
        arange_w = (torch.arange(0, self.W)
            .unsqueeze(0).unsqueeze(0)
            .to(self.kernel_projection1.weight.device)
        )
        self.register_buffer('arange_h_buf', arange_h)
        self.register_buffer('arange_w_buf', arange_w)
        
        # norm
        self.norm_dict[f'offset_ln_feats_coords'] = LayerNormAffineXC(
            self.params.offset_lstm_dim, 
            (self.num_nodes, self.params.offset_lstm_dim)
        )
        self.norm_dict[f'offset_ln_lstm'] = LayerNormAffineXC(
            self.params.offset_lstm_dim,
            (self.num_nodes, self.params.offset_lstm_dim)
        )

        if self.project_i3d :
            self.norm_dict[f'dynamic_graph_projection'] = LayerNormAffine2D(
                self.C, (self.C,self.H,self.W) )

        self.aux_feats = {}

    def apply_norm(self, input, name, zero_init=False):
        # input: B*T x N x C
        return self.norm_dict[name](input)

    def create_offset_modules(self):
        # for fullresolution evaluation: pool the offset_generation input
        offset_input_size = self.H
        if self.params.full_res:
            # TODO: de schimbat hardcodarea
            self.input_pooling = nn.AdaptiveAvgPool2d((self.iH,self.iW))
            offset_input_size = self.iH
    
        if self.params.offset_generator == 'big':
            self.fishnet = Fishnet(input_channels=self.C, input_height=offset_input_size)
            full_res_pad = 0
            # TODO: hack, at training time the global conv acts as a fully connected
            # for fullsize evaluation it should be the same size as at training
            global_conv_size = self.fishnet.norm_size[-1] - full_res_pad
            
            self.global_conv = nn.Sequential(
                nn.Conv2d(self.fishnet.offset_channels[-1], self.num_nodes * self.params.offset_lstm_dim, 
                        [global_conv_size, global_conv_size]),
                nn.AdaptiveAvgPool2d((1,1))
            )

        elif self.params.offset_generator == 'small':
            self.glore = GloRe(input_channels=self.C, input_height=self.H, h=self.h, w=self.w)

        
        self.offset_lstm = recurrent_net(batch_first=True,
            input_size=self.params.offset_lstm_dim, hidden_size=self.params.offset_lstm_dim)
        

        self.offset_pred =  torch.nn.Linear(self.params.offset_lstm_dim, 4)
        self.offset_alpha = torch.nn.Parameter(torch.zeros(size=[4], requires_grad=True))

    def compute_anchors(self):
        if self.params.init_regions == 'center':
            ph = self.H / 2 * torch.ones(1,self.h, 1)
            pw = self.H / 2 * torch.ones(1,self.w, 1)
            # ph: BT x (h * w) x 1
            ph = ph.repeat(1, 1, self.w).view(1, self.h * self.w, 1)
            # ph: BT x (h * w) x 1
            pw = pw.repeat(1, self.h, 1)

        elif self.params.init_regions == 'grid':
            ph = torch.linspace(0, self.H, 2 * self.h + 1)[1::2]
            ph = ph.unsqueeze(-1).unsqueeze(0)

            pw = torch.linspace(0, self.W, 2 * self.w + 1)[1::2]
            pw = pw.unsqueeze(-1).unsqueeze(0)
            ph = ph.repeat(1, 1, self.w).view(1, self.h * self.w, 1)
            #ph: BT x (h * w) x 1
            pw = pw.repeat(1, self.h, 1)
        else:
            print(f'init_regions: center or grid')
            sys.exit()
        return ph, pw


    def get_offsets(self, input_features, name='offset'):
        # resize the input for the offset generation functions
        if self.params.full_res:
            input_features = self.input_pooling(input_features)
        # input_features: B * T, C, H, W 
        # get global input embedding
        if self.params.offset_generator == 'big':
            global_features = self.fishnet(input_features)
            assert global_features.shape[-1] == self.fishnet.norm_size[-1] and global_features.shape[-2] == self.fishnet.norm_size[-1]
            # assert global_features.shape[-1] == 5 and global_features.shape[-2] == 5

            global_features = self.global_conv(global_features)
            global_features = global_features.view(self.BT, self.num_nodes, self.params.offset_lstm_dim)

        elif self.params.offset_generator == 'small':
            global_features = self.glore(input_features)
            global_features = global_features.permute(0,2,1)

        # norm
        global_features = self.apply_norm(global_features, f'{name}_ln_feats_coords')
        global_features = global_features.view(self.B, self.T, self.num_nodes, self.params.offset_lstm_dim)
        global_features = global_features.permute(0,2,1,3).contiguous()
        global_features = global_features.view(self.B * self.num_nodes, self.T, self.params.offset_lstm_dim)
        
        # aplly recurrence
        self.offset_lstm.flatten_parameters()
        offset_feats, _ = self.offset_lstm(global_features)
        offset_feats = offset_feats.view(self.B, self.num_nodes, self.T, self.params.offset_lstm_dim)
        offset_feats = offset_feats.permute(0,2,1,3).contiguous()
        offset_feats = offset_feats.view(self.BT, self.num_nodes, self.params.offset_lstm_dim)
        offset_feats = self.apply_norm(offset_feats, f'{name}_ln_lstm')

        # predict final coordinates
        offsets = self.offset_pred(offset_feats)
        offsets = offsets * self.offset_alpha
        return offsets

    def get_dynamic_kernel(self, offsets):
        offsets = offsets.view(offsets.shape[0],-1,4)
        # create the fixed anchors
        ph = self.ph_buf
        pw = self.pw_buf
        self.arange_h = self.arange_h_buf
        self.arange_w = self.arange_w_buf

        # pw: BT x (h * w) x 1
        # scale the offsets so that they always stay in the input region
        regions_h = torch.exp(offsets[:,:,0]).unsqueeze(-1)
        regions_w = torch.exp(offsets[:,:,1]).unsqueeze(-1)

        regions_dh = torch.tanh(offsets[:,:,2].unsqueeze(-1) + atanh( 2 * ph / self.H - 1))
        regions_dw = torch.tanh(offsets[:,:,3].unsqueeze(-1) + atanh( 2 * pw / self.W - 1))


        h_init = self.H / (2 * self.h) + 1
        w_init = self.W / (2 * self.w) + 1

        regions_h = regions_h * h_init + 1
        regions_w = regions_w * w_init + 1

        regions_dh = regions_dh * (self.H / 2) + self.H / 2 - ph
        regions_dw = regions_dw * (self.W / 2) + self.W / 2 - pw



        # B*T x N x H x 1
        kernel_center_h = torch.clamp(ph + regions_dh, 0, self.H - 1)
        dist_h = torch.abs(self.arange_h - kernel_center_h)
        kernel_h = F.relu(regions_h - dist_h)
        kernel_h = kernel_h.unsqueeze(-1)
        # B*T x N x 1 x W
        kernel_center_w = torch.clamp(pw + regions_dw, 0, self.W - 1)
        dist_w = torch.abs(self.arange_w - kernel_center_w)
        kernel_w = F.relu(regions_w - dist_w)
        kernel_w = kernel_w.unsqueeze(-2)
        # B*T x N x H x W
        kernel = torch.matmul(kernel_h, kernel_w)
        
        self.offsets = torch.stack([regions_h, regions_w, kernel_center_h, kernel_center_w], dim=3)

        eps = 1e-37
        kernel = kernel / (torch.sum(kernel, dim=(2,3), keepdims=True)  + eps)
        # B*T x h x w x H x W
        kernel = kernel.view(self.BT, self.h, self.w, self.H, self.W)
        
        # necessary to compute the bounding box
        self.kernel_center_h = kernel_center_h
        self.kernel_center_w = kernel_center_w
        self.kernel_h = regions_h
        self.kernel_w = regions_w
        return kernel

    def set_input(self, input_features):
        # input_features: B * T, C, H, W 
        patch_feats = []
        nodes_wo_position = []
        # # tmp
        for idx, scale in enumerate([1]): # TODO or NOT TODO:multiscale

            if self.params.dynamic_regions == 'none':
                pass
            elif self.params.dynamic_regions in ['pos_only', 'dyreg','semantic']:
                # BT x 9 x 4
                if self.params.dynamic_regions == 'semantic':
                    self.kernel = self.get_glore_kernel(input_features)
                    self.offsets = self.kernel # TODO: de forma, doar ca sa pastram interfata din models.py
                    
                else:
                    offsets = self.get_offsets(input_features)
                    self.kernel = self.get_dynamic_kernel(offsets)
                    initial_kernel = self.kernel

            act_scale = differentiable_resize_area(input_features, self.kernel)
            act_scale = act_scale.view(self.B, self.T, self.C, self.h * self.w)

            # add location embeding to each node
            # B x T x num_nodes x (H*W)
            kernel_location = initial_kernel.view(self.B, self.T, self.h * self.w, self.H * self.W)
            # for fullsize evaluation the kernel should have the same shape as it did at training time
            if self.params.full_res:
                kernel_location = initial_kernel.view(self.BT, self.h * self.w, self.H, self.W)
                kernel_location = self.kernel_location_pool(kernel_location)
                kernel_location = kernel_location.view(self.B, self.T, self.h * self.w, self.iH * self.iW)

            kernel_location = self.kernel_projection1(kernel_location)
            kernel_location = kernel_location.permute(0,1,3,2)

            nodes_wo_position.append(act_scale)
            act_scale = act_scale + kernel_location

            patch_feats.append(act_scale)
        # nodes: B x T x C x N
        nodes = torch.cat(tuple(patch_feats), dim=3)

        self.nodes_wo_position = torch.cat(tuple(nodes_wo_position), dim=3)
        self.nodes = nodes
        # self.kernel:  BT x 3 x 3 x 28 x 28
        self.kernel_foreground = self.kernel.mean(dim=1, keepdim=True).mean(dim=2, keepdim=True)
        kernel_background = self.kernel_foreground.max(dim=-1, keepdim=True)[0].max(dim=-2, keepdim=True)[0] - self.kernel_foreground
        self.kernel_background = kernel_background / kernel_background.sum(dim=-1, keepdim=True).sum(dim=-2, keepdim=True)
        self.kernel_foreground = self.kernel_foreground / self.kernel_foreground.sum(dim=-1, keepdim=True).sum(dim=-2, keepdim=True)

        self.global_mean_features = input_features.mean(dim=-1).mean(dim=-1)
        self.nodes_background = differentiable_resize_area(input_features, self.kernel_background)
        self.nodes_foreground = self.nodes.mean(dim=-1)

        self.aux_feats['global_mean_features'] = self.global_mean_features
        self.aux_feats['nodes_background'] = self.nodes_background
        self.aux_feats['nodes_foreground'] = self.nodes_foreground

        self.kernel_foreground = self.kernel_foreground.view(self.kernel_foreground.shape[0], -1)
        self.aux_feats['kernel_foreground'] = self.kernel_foreground
        return nodes

    # project features from graph to features map: RSTG_to_map model    
    def remap_nodes(self, nodes):
        # nodes: B x T x C x num_nodes
        start = end = out = 0
        #for scale in self.gc.scales:
        for scale in [(self.h,self.w)]:
        
            end += scale[0] * scale[1]
            nodes_scale = nodes[:,:,:,start:end]
            nodes_scale = nodes_scale.view(self.BT, nodes.shape[2], scale[0], scale[1])
            start += scale[0] * scale[1]

            if self.params.dynamic_regions != 'none':    
                kernel = self.kernel.permute(0,3,4,1,2)

            out_scale = differentiable_resize_area( nodes_scale, kernel)
        
            out = out + out_scale

        return out

    def forward(self, input_feats):
        # input_feats: B x T x C x H x W -> BT x C x H x W
        # TODO: add B, T as parameters for dyreg
        self.B = input_feats.shape[0]
        self.T = input_feats.shape[1]
        self.BT = self.B * self.T
        input_feats = input_feats.view(self.B * self.T, *input_feats.shape[2:])


        if self.project_i3d: # and self.params.project_at_input:
            x = self.project_i3d_linear(input_feats)
            x = self.apply_norm(x, 'dynamic_graph_projection')
        else:
            x = input_feats
        nodes = self.set_input(x)
        # nodes: B x T x C x N
        nodes = self.rstg(nodes)
        # nodes: B x T x C x N
       
        out = self.remap_nodes(nodes)
        if self.project_i3d:# and self.params.project_at_input and self.params.remap_first==True:
            out = out.view(self.B*self.T, self.C, self.H, self.W)
            out = self.project_back_i3d_linear(out)

        # out: B x T x C x H x W
        out = out.view(self.B, self.T, *out.shape[1:])

        return out


class DynamicGraphWrapper(nn.Module):
    def __init__(self, params , 
            H=14, W=14, oH=7, oW=7, iH=None, iW=None,
            h=3, w=3,
            node_dim=512, in_channels=1024, out_num_ch = 2048,
            project_i3d=False, name=''):
        super(DynamicGraphWrapper, self).__init__()
        # self.block = block
        self.dynamic_graph = DynamicGraph(params=params, backbone_dim=in_channels,
            H=H,W=W,oH=oH, oW=oW, iH=iH, iW=iW,
            h=h,w=w,
            node_dim=node_dim, out_num_ch =out_num_ch,
            project_i3d=project_i3d, name=name)
        self.params = params
        self.in_channels = in_channels
        self.H = H
        self.W = W
        # the dimenstion of the input used at training time
        # to be used for resize for fullsize evaluation
        self.iH = iH
        self.iW = iW

        # residual branch
        # for paraleel branch, initialize the residual branch with 0, 
        # so that it is ignored at the begening of the optimization
        self.norm_dict = nn.ModuleDict({})
        if self.params.rstg_combine == 'serial':
            zero_init = False
        else:
            zero_init = True
            self.norm_dict['residual_norm'] = LayerNormAffine2D(self.in_channels, (self.in_channels, self.H, self.W),
                zero_init=zero_init)


    def apply_norm(self, input, name):
        # input: B*T x C x H x W
        return self.norm_dict[name](input)

    def forward(self, x):
        y = self.dynamic_graph(x)
        y = y.view(y.shape[0] * y.shape[1], *y.shape[2:])
        y = self.apply_norm(y, 'residual_norm')

        if self.params.rstg_combine == 'serial':
            out = y
        elif self.params.rstg_combine == 'plus':
            out = x.view(*y.shape) + y

        return out


def make_rstg(net):
    places = args.place_graph.replace('layer','').split('_')
    
    graph_params = args.graph_params
    out_pool_size = args.out_pool_size
    out_num_ch = args.out_num_ch
    
    print(f'out_pool_size: {out_pool_size}')

    list_layer = {}
    list_layer[1] = [x for x in net.layer1]
    list_layer[2] = [x for x in net.layer2]
    list_layer[3] = [x for x in net.layer3]
    list_layer[4] = [x for x in net.layer4]

    dyreg_params = dyregParams()
    dyreg_params.set_from_args(args)
   
    for place in places:
        layer = int(place.split('.')[0])
        block = int(place.split('.')[1])
        if args.bottleneck_graph:
            list_layer[layer][block].conv2 = nn.Sequential(
                SplitBT(block=list_layer[layer][block].conv2, T=16),
                DynamicGraphWrapper(dyreg_params,
                # list_layer[layer][block].conv2 = DynamicGraphWrapper(list_layer[layer][block].conv2, 
                in_channels=graph_params[layer]['in_channels'], 
                H=graph_params[layer]['H'], W=graph_params[layer]['H'], 
                oH=out_pool_size, oW=out_pool_size, 
                iH=graph_params[layer]['iH'], iW=graph_params[layer]['iH'],
                out_num_ch = out_num_ch,
                node_dim=graph_params[layer]['node_dim'], project_i3d=graph_params[layer]['project_i3d'],
                name=graph_params[layer]['name'])
            )
        else:
            list_layer[layer][block] = DynamicGraphWrapper(dyreg_params, 
                            in_channels=graph_params[layer]['in_channels'], 
                            H=graph_params[layer]['H'], W=graph_params[layer]['H'], 
                            oH=out_pool_size, oW=out_pool_size, 
                            out_num_ch = out_num_ch,
                            node_dim=graph_params[layer]['node_dim'], project_i3d=graph_params[layer]['project_i3d'],
                            name=graph_params[layer]['name'])

        net.layer1 = nn.Sequential(*list_layer[1])
        net.layer2 = nn.Sequential(*list_layer[2])
        net.layer3 = nn.Sequential(*list_layer[3])
        net.layer4 = nn.Sequential(*list_layer[4])
    

def save_image(out, name='img'):
    out = out.permute(0,2,3,4,1)
    out_image = out.detach().numpy()[0,0]
    out_image = out_image - out_image.min()
    out_image = out_image / out_image.max()
    f, axarr = plt.subplots(1,1)
    axarr.imshow(out_image) 
    plt.savefig(name+'.jpeg') 

class SplitBT(nn.Module):
    def __init__(self, block, T):
        super(SplitBT, self).__init__()

        self.block = block
        self.T = T

    def forward(self, x):
        x = self.block(x)
        B = x.shape[0] // self.T
        out = x.view(B, self.T, *x.shape[1:])
        return out

class dyregParams(nn.Module):
    def __init__(self):
        super(dyregParams, self).__init__()

        self.full_res = False
        self.rstg_combine = 'plus'
        self.dynamic_regions = 'dyreg'
        self.offset_lstm_dim = 128
        self.offset_generator = 'big'
        self.init_regions = 'center'
        self.combine_by_sum = True
        self.aggregation_type = 'dot'
        self.send_layers = 1
        self.update_layers = 0

    def set_from_args(self, args):
        self.full_res = args.full_res
        self.rstg_combine = args.rstg_combine
        self.dynamic_regions = args.dynamic_regions
        self.offset_lstm_dim = args.offset_lstm_dim
        self.offset_generator = args.offset_generator
        self.init_regions = args.init_regions
