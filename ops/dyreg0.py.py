# Non-local block using embedded gaussian
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
import time
from opts import parser
from ops.rstg import *
from ops.models_utils import *

global args, best_prec1
from opts import parse_args
args = parser.parse_args()
args = parse_args()

import pickle
import pdb



class DynamicGraph(nn.Module):
    def __init__(self, 
            backbone_dim=1024, 
            H=14, W=14,oH=7,oW=7,iH=None,iW=None,
            h=3,w=3,
            node_dim=512, out_num_ch = 2048, project_i3d=False, name=''):
        super(DynamicGraph, self).__init__()
        self.params = args
        # the dimenstion of the input used at training time
        # to be used for resize for fullsize evaluation
        self.iH = iH
        self.iW = iW
        self.name = name
        self.rstg = RSTG(backbone_dim=backbone_dim, node_dim=node_dim, project_i3d=project_i3d)

        if not args.distributed:
            self.B = (self.params.test_crops * self.params.batch_size 
                // max(1,torch.cuda.device_count()) 
                * (2 if self.params.twice_sample else 1)
            )
        else:
            self.B = self.params.batch_size 
            
        # self.T = self.params.rstg_time_steps
        self.T = self.params.num_segments
        self.BT = self.B * self.T
        self.backbone_dim = backbone_dim
        if self.params.project_at_input:
            self.C = self.rstg.node_dim
        else:
            self.C = backbone_dim
        # self.W = self.H = 14
        self.out_num_ch = out_num_ch
        self.H = H
        self.W = W
        self.oH = oH
        self.oW = oW
        self.h = h
        self.w = w
        self.num_nodes = self.h * self.w
        self.project_i3d = project_i3d
        # self.initial_offsets = torch.zeros(self.BT, self.h, self.w, 4)
        # get anchors
        ph, pw = self.compute_anchors()
        self.register_buffer('ph_buf', ph)
        self.register_buffer('pw_buf', pw)

        self.norm_dict = nn.ModuleDict({})
        # input projection           

        if self.project_i3d and  self.params.project_at_input:
            self.project_i3d_linear = nn.Conv2d(backbone_dim, self.C, [1, 1])
            if self.params.remap_first == False:
                self.project_back_i3d_linear = nn.Conv1d(self.C, backbone_dim, [1])
            else:
                self.project_back_i3d_linear = nn.Conv2d(self.C, backbone_dim, [1,1])

        if self.params.rstg_skip_connection:
            self.project_skip_graph = nn.Conv2d(node_dim,
                self.out_num_ch//args.ch_div, [1, 1]
            )
            self.norm_dict[f'skip_conn'] = LayerNormAffine2D(
                self.out_num_ch//args.ch_div, 
                (self.out_num_ch//args.ch_div,self.H,self.W)
            )
            self.avg_pool_7 = nn.AdaptiveAvgPool2d((self.oH,self.oW))
    
        # not use otherwise but created already so this is to avoid resume errors
        if self.params.dynamic_regions == 'constrained_fix_size':
            # create kernel constants
            # const_dh_ones = torch.ones(self.BT, self.num_nodes, 1)
            # const_dw_ones = torch.ones(self.BT, self.num_nodes, 1)
            const_dh_ones = torch.ones(1, self.num_nodes, 1)
            const_dw_ones = torch.ones(1, self.num_nodes, 1)
            self.register_buffer('const_dh_ones_buf', const_dh_ones)
            self.register_buffer('const_dw_ones_buf', const_dw_ones)
        
        if self.params.dynamic_regions == 'GloRe':
            self.mask_fishnet = Fishnet(
                input_channels=self.C, input_height=self.H, keep_spatial_size=True
            )
            self.mask_pred = nn.Conv2d(self.C, self.num_nodes, [1, 1])
        
        
        self.create_offset_modules()
        # project kernel
        if args.full_res in [
            "resize_offset_generator_output", "resize_offset_generator_input",
        ]:
            self.kernel_projection1 = torch.nn.Linear(self.iH * self.iW, self.C)
            self.kernel_location_pool = nn.AdaptiveAvgPool2d((self.iH,self.iW))
        else:
            self.kernel_projection1 = torch.nn.Linear(self.H * self.W, self.C) 
        
        if not self.params.combine_by_sum:
            self.kernel_projection2 = torch.nn.Linear(2 * self.C, self.C) 
        
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

        if args.tmp_norm_before_dynamic_graph:
            self.norm_dict[f'before_dynamic_graph'] = LayerNormAffine2D(
                backbone_dim, 
                (backbone_dim,self.H,self.W)
            )
        if (self.project_i3d and self.params.project_at_input 
                and args.tmp_norm_after_project_at_input):
            self.norm_dict[f'dynamic_graph_projection'] = LayerNormAffine2D(
                self.C, (self.C,self.H,self.W) )

        if self.params.fix_offsets:
            fix_offsets = torch.zeros(size=[self.BT,self.num_nodes, 4], requires_grad=False)
            self.register_buffer('fix_offsets', fix_offsets)
        # fix kernel
        if self.params.dynamic_regions == 'none':
        # if True:
            fix_kernel = self.get_fix_kernel()
            self.register_buffer('fix_kernel', fix_kernel)
            self.fix_kernel = self.fix_kernel.unsqueeze(0).repeat(self.BT,1,1,1,1)
        if self.params.node_confidence == 'pool_feats':
            self.node_confidence = nn.Conv1d(self.C, 1, [1])

        if self.params.contrastive_mlp == True:
            self.contrastive_mlp = nn.Sequential(
                nn.Linear(self.C, self.C),
                nn.ReLU(),
                nn.Linear(self.C, self.C)
            )

        self.aux_feats = {}
    def get_norm(self, input, name, zero_init=False):
        # input: B*T x N x C
        return self.norm_dict[name](input)

    def create_offset_modules(self):
        # for fullresolution evaluation: pool the offset_generation input
        offset_input_size = self.H
        if self.params.full_res == 'resize_offset_generator_input':
            # TODO: de schimbat hardcodarea
            self.params.input_size = 256
            self.input_pooling = nn.AdaptiveAvgPool2d((self.iH,self.iW))
            offset_input_size = self.iH
    
        if self.params.offset_generator == 'fishnet':
            self.fishnet = Fishnet(input_channels=self.C, input_height=offset_input_size)
            full_res_pad = 0
            if args.full_res == 'resize_offset_generator_output':
                full_res_pad = 1
            # TODO: hack, at training time the global conv acts as a fully connected
            # for fullsize evaluation it should be the same size as at training
            global_conv_size = self.fishnet.norm_size[-1] - full_res_pad
            
            self.global_conv = nn.Sequential(
                nn.Conv2d(self.fishnet.offset_channels[-1], self.num_nodes * self.params.offset_lstm_dim, 
                        [global_conv_size, global_conv_size]),
                nn.AdaptiveAvgPool2d((1,1))
            )

        elif self.params.offset_generator == 'glore':
            self.glore = GloRe(input_channels=self.C, input_height=self.H, h=self.h, w=self.w)

        elif self.params.offset_generator == 'fishnet-glore':
            self.fishnet = Fishnet(input_channels=self.C, input_height=self.H)
            if self.H < 14:
                glore_input_height = 5
            else:
                glore_input_height = 6
            self.glore = GloRe(input_channels=16, input_height=glore_input_height, h=self.h, w=self.w)
        
        self.offset_lstm = recurrent_net(batch_first=True,
            input_size=self.params.offset_lstm_dim, hidden_size=self.params.offset_lstm_dim)
        
        if self.params.node_confidence == 'offset':
            # the first 4 offsets are used for region location and size
            # the 5-th offset is used as region confidence
            self.offset_pred =  torch.nn.Linear(self.params.offset_lstm_dim, 5)
            init_alpha1 = torch.zeros(size=[4], requires_grad=True)
            init_alpha2 = torch.ones(size=[1], requires_grad=True)
            init_alpha = torch.cat((init_alpha1, init_alpha2),dim=-1)
            self.offset_alpha = torch.nn.Parameter(init_alpha)
        else:
            self.offset_pred =  torch.nn.Linear(self.params.offset_lstm_dim, 4)
            self.offset_alpha = torch.nn.Parameter(torch.zeros(size=[4], requires_grad=True))

    def compute_anchors(self):
        if self.params.init_regions == 'center':
            ph = self.H / 2 * torch.ones(1,self.h, 1)
            pw = self.H / 2 * torch.ones(1,self.w, 1)
            # ph = ph.repeat(self.BT, 1, self.w).view(self.BT, self.h * self.w, 1)
            # # ph: BT x (h * w) x 1
            # pw = pw.repeat(self.BT, self.h, 1)
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
        if self.params.full_res == 'resize_offset_generator_input':
            input_features = self.input_pooling(input_features)
        # input_features: B * T, C, H, W 
        if self.params.offset_generator == 'fishnet':
            global_features = self.fishnet(input_features)
            assert global_features.shape[-1] == self.fishnet.norm_size[-1] and global_features.shape[-2] == self.fishnet.norm_size[-1]
            # assert global_features.shape[-1] == 5 and global_features.shape[-2] == 5

            global_features = self.global_conv(global_features)
            global_features = global_features.view(self.BT, self.num_nodes, self.params.offset_lstm_dim)

        elif self.params.offset_generator == 'glore':
            global_features = self.glore(input_features)
            global_features = global_features.permute(0,2,1)

        elif self.params.offset_generator == 'fishnet-glore':
            # print(f"Input fihsnet {input_features.shape}")
            global_features = self.fishnet(input_features)
            # print(f"Output fishnet {global_features.shape}")
            global_features = self.glore(global_features)
            global_features = global_features.permute(0,2,1)

        global_features = self.get_norm(global_features, f'{name}_ln_feats_coords')
        global_features = global_features.view(self.B, self.T, self.num_nodes, self.params.offset_lstm_dim)
        global_features = global_features.permute(0,2,1,3).contiguous()
        global_features = global_features.view(self.B * self.num_nodes, self.T, self.params.offset_lstm_dim)
        
        self.offset_lstm.flatten_parameters()
        offset_feats, _ = self.offset_lstm(global_features)
        offset_feats = offset_feats.view(self.B, self.num_nodes, self.T, self.params.offset_lstm_dim)
        offset_feats = offset_feats.permute(0,2,1,3).contiguous()
        offset_feats = offset_feats.view(self.BT, self.num_nodes, self.params.offset_lstm_dim)
        offset_feats = self.get_norm(offset_feats, f'{name}_ln_lstm')

        offsets = self.offset_pred(offset_feats)
        offsets = offsets * self.offset_alpha
        # if args.freeze_regions in ['offsets_input', 'just_offsets'] :
        #     offsets = offsets.detach()
        return offsets

    def get_fix_kernel(self):
        import numpy as np
        saved_filters_name = f'/root/experimantal/dynamic_graphs_5aug/dynamic_graph_regions/util_pickles/image_resize_area_align_false_filters_HWhw_{self.H}_{self.W}_{self.h}_{self.w}.pickle'
        print(saved_filters_name)
        if os.path.exists(saved_filters_name):
            with open(saved_filters_name, 'rb') as handle:
                saved_filters = pickle.load(handle)
        filters = saved_filters['filters'].astype(np.float32)
        kernel = torch.tensor(filters, requires_grad=False)
        return kernel
    
    def get_glore_kernel(self, input_features):
        global_features = self.mask_fishnet(input_features)
        mask = self.mask_pred(global_features)
        mask = mask.view(mask.shape[0], self.h, self.h, *mask.shape[2:])
        return mask

    def get_dynamic_kernel(self, offsets):
        offsets = offsets.view(offsets.shape[0],-1,4)
        # create the fixed anchors
        # ph = self.ph
        # pw = self.pw
        ph = self.ph_buf
        pw = self.pw_buf
        # ph = self.ph_buf.repeat((self.BT,1,1))
        # pw = self.pw_buf.repeat((self.BT,1,1))
        self.arange_h = self.arange_h_buf
        self.arange_w = self.arange_w_buf

        # pw: BT x (h * w) x 1
        # scale the offsets so that they always stay in the input region

        if self.params.use_detector:
            # TODO: poate separat pe H, W
            # print(f'input_size: {self.params.input_size}')
            offsets = offsets / self.params.input_size * self.H
            # offsets: w1 h1 w2 h2
            # offsets: h1 w1 h2 w2
            regions_h = (offsets[:,:,2].unsqueeze(-1) - offsets[:,:,0].unsqueeze(-1) ) / 2.0
            regions_w = (offsets[:,:,3].unsqueeze(-1) - offsets[:,:,1].unsqueeze(-1) ) / 2.0

            regions_dh = (offsets[:,:,0].unsqueeze(-1) + offsets[:,:,2].unsqueeze(-1) ) / 2.0 - ph
            regions_dw = (offsets[:,:,1].unsqueeze(-1) + offsets[:,:,3].unsqueeze(-1) ) / 2.0 - pw

            if self.params.gt_detection_mult > 1.0:
                regions_h = regions_h * self.params.gt_detection_mult
                regions_w = regions_w * self.params.gt_detection_mult
    
        elif self.params.dynamic_regions == 'constrained':
            regions_h = torch.exp(offsets[:,:,0]).unsqueeze(-1)
            regions_w = torch.exp(offsets[:,:,1]).unsqueeze(-1)

            regions_dh = torch.tanh(offsets[:,:,2].unsqueeze(-1) + atanh( 2 * ph / self.H - 1))
            regions_dw = torch.tanh(offsets[:,:,3].unsqueeze(-1) + atanh( 2 * pw / self.W - 1))

            if args.tmp_init_different != 0.0:
                h_init = self.H / (args.tmp_init_different*self.h) 
                w_init = self.W / (args.tmp_init_different*self.w) 
            else:
                h_init = self.H / (2 * self.h) + 1
                w_init = self.W / (2 * self.w) + 1


            regions_h = regions_h * h_init
            regions_w = regions_w * w_init 

            regions_dh = regions_dh * (self.H / 2) + self.H / 2 - ph
            regions_dw = regions_dw * (self.W / 2) + self.W / 2 - pw

            if self.params.kernel_type == 'gaussian':
                regions_h = regions_h / 2.2
                regions_w = regions_w / 2.2
            else:
                if self.params.tmp_increment_reg and args.tmp_init_different == 0.0:
                    regions_h = regions_h + 1
                    regions_w = regions_w + 1

            if self.params.region_size_boost != 1.0 and self.params.region_size_boost != None:
                regions_h = regions_h * self.params.region_size_boost
                regions_w = regions_w * self.params.region_size_boost

        elif self.params.dynamic_regions == 'constrained_fix_size':
            self.const_dh_ones = self.const_dh_ones_buf.repeat([self.BT,1,1])
            self.const_dw_ones = self.const_dw_ones_buf.repeat([self.BT,1,1])

            regions_dh = torch.tanh(offsets[:,:,2].unsqueeze(-1) + atanh( 2 * ph / self.H - 1))
            regions_dw = torch.tanh(offsets[:,:,3].unsqueeze(-1) + atanh( 2 * pw / self.W - 1))

            regions_dh = regions_dh * (self.H / 2) + self.H / 2 - ph
            regions_dw = regions_dw * (self.W / 2) + self.W / 2 - pw



            regions_h = self.const_dh_ones * (self.H / (2*self.h) + 1)
            regions_w = self.const_dw_ones * (self.W / (2*self.w) + 1)

            if self.params.use_region_isc:
                regions_h = self.const_dh_ones * (self.H / self.params.region_isc + 1)
                regions_w = self.const_dw_ones * (self.W / self.params.region_isc + 1)
            if self.params.kernel_type == 'gaussian':
                regions_h = regions_h / 2.2
                regions_w = regions_w / 2.2
            # regions_dh: BT x N x 1
            # regions_dw: BT x N x 1
        else:
            print('select correct dynamic_regions params')
            sys.exit()

        # B*T x N x H x 1
        kernel_center_h = torch.clamp(ph + regions_dh, 0, self.H - 1)
        if self.params.kernel_type == 'gaussian':
            dist_h = (self.arange_h - kernel_center_h) / regions_h
            kernel_h = dist_h * dist_h
            kernel_h = torch.exp(- 1.0 / 2.0 * kernel_h)
        else:
            dist_h = torch.abs(self.arange_h - kernel_center_h)
            kernel_h = F.relu(regions_h - dist_h)
        kernel_h = kernel_h.unsqueeze(-1)
        # B*T x N x 1 x W
        kernel_center_w = torch.clamp(pw + regions_dw, 0, self.W - 1)
        if self.params.kernel_type == 'gaussian':
            dist_w = (self.arange_w - kernel_center_w) / regions_w
            kernel_w = dist_w * dist_w
            kernel_w = torch.exp(- 1.0 / 2.0 * kernel_w)
        else:
            dist_w = torch.abs(self.arange_w - kernel_center_w)
            kernel_w = F.relu(regions_w - dist_w)
        kernel_w = kernel_w.unsqueeze(-2)
        # B*T x N x H x W
        kernel = torch.matmul(kernel_h, kernel_w)
        
        self.offsets = torch.stack([regions_h, regions_w, kernel_center_h, kernel_center_w], dim=3)

        eps = 1e-37
        kernel = kernel / (torch.sum(kernel, dim=(2,3), keepdims=True)  + eps)
        # kernel = (kernel == kernel.max(2)[0].max(2)[0].unsqueeze(2).unsqueeze(2)).float()
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
        for idx, scale in enumerate([1]): # TODO or NOT TODO:multiscale

            if self.params.dynamic_regions == 'none':
                pass
            elif self.params.dynamic_regions in [
                    'constrained_fix_size', 'constrained',
                    'free', 'free_shift', 'constrained_size1','GloRe'
                ]:
                # BT x 9 x 4
                if self.params.dynamic_regions == 'GloRe':
                    self.kernel = self.get_glore_kernel(input_features)
                    self.offsets = self.kernel # TODO: de forma, doar ca sa pastram interfata din models.py
                    
                else:
                    offsets = self.get_offsets(input_features)
                    
                    self.node_conf = 1
                    if self.params.node_confidence == 'offset':
                        self.node_conf = F.sigmoid(offsets[:,:,4:5]) 
                        self.node_conf = self.node_conf.view(self.node_conf.shape[0], self.h, self.w,1,1)
                        offsets_param = offsets[:,:,:4]
                        self.kernel = self.get_dynamic_kernel(offsets_param)
                        initial_kernel = self.kernel
                        self.kernel = self.node_conf * self.kernel

                    else:    
                        self.kernel = self.get_dynamic_kernel(offsets)
                        initial_kernel = self.kernel

            act_scale = differentiable_resize_area(input_features, self.kernel)
            act_scale = act_scale.view(self.B, self.T, self.C, self.h * self.w)
            if self.params.node_confidence == 'pool_feats': 
                act_view = act_scale.view(act_scale.shape[0] * act_scale.shape[1],
                    act_scale.shape[2], act_scale.shape[3]
                )
                self.node_conf = self.node_confidence(act_view)
                self.node_conf = F.sigmoid(self.node_conf)
                self.node_conf = self.node_conf.view(act_scale.shape[0], 
                    act_scale.shape[1], self.node_conf.shape[1], self.node_conf.shape[2]
                )
                act_scale = act_scale * self.node_conf

            # add location embeding to each node
            # B x T x num_nodes x (H*W)
            kernel_location = initial_kernel.view(self.B, self.T, self.h * self.w, self.H * self.W)

            if args.tmp_fix_kernel_grads:
                # stop gradient
                kernel_location = kernel_location.detach()

            # for fullsize evaluation the kernel should have the same shape as it did at training time
            if (args.full_res in ['resize_offset_generator_output', 'resize_offset_generator_input']):
                kernel_location = initial_kernel.view(self.BT, self.h * self.w, self.H, self.W)
                kernel_location = self.kernel_location_pool(kernel_location)
                kernel_location = kernel_location.view(self.B, self.T, self.h * self.w, self.iH * self.iW)


            kernel_location = self.kernel_projection1(kernel_location)
            kernel_location = kernel_location.permute(0,1,3,2)

            if not args.tmp_fix_kernel_grads:
                # stop gradient
                kernel_location = kernel_location.detach()
            # stop gradient
            # kernel_location = kernel_location.detach()

            nodes_wo_position.append(act_scale)
            if self.params.combine_by_sum:
                act_scale = act_scale + kernel_location
            else:
                act_scale = torch.cat((act_scale, kernel_location), dim=2)

                act_scale = act_scale.permute(0,1,3,2)
                act_scale = self.kernel_projection2(act_scale)
                act_scale = act_scale.permute(0,1,3,2)

            patch_feats.append(act_scale)
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

        if self.params.contrastive_mlp == True:
            self.global_mean_features = self.contrastive_mlp(self.global_mean_features)

            self.nodes_background = self.nodes_background.squeeze(dim=-1).squeeze(dim=-1)
            self.nodes_background = self.contrastive_mlp(self.nodes_background)
            self.nodes_background = self.nodes_background.unsqueeze(dim=-1).unsqueeze(dim=-1)

            self.nodes_foreground = self.contrastive_mlp(self.nodes_foreground)

            self.nodes_wo_position = self.nodes_wo_position.permute(0,1,3,2)
            self.nodes_wo_position = self.contrastive_mlp(self.nodes_wo_position)
            self.nodes_wo_position = self.nodes_wo_position.permute(0,1,3,2)


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

        # out: B x T x dim_nodes x H x W
        if self.params.rstg_skip_connection:
            aux_out = self.project_skip_graph(out)
            aux_out = self.get_norm(aux_out, 'skip_conn', zero_init=False)
            self.graph_out_map = self.avg_pool_7(aux_out)
        return out

    def get_gernel(self):
        return self.dynamic_graph.kernel
    def forward(self, input_feats):

        if args.tmp_norm_before_dynamic_graph:  
            input_feats = self.get_norm(input_feats, 'before_dynamic_graph')
        # x: B*T x C x H x W
        if self.project_i3d and self.params.project_at_input:
            x = self.project_i3d_linear(input_feats)
            if args.tmp_norm_after_project_at_input:
                x = self.get_norm(x, 'dynamic_graph_projection')
        else:
            x = input_feats
        nodes = self.set_input(x)
        # nodes: B x T x C x N
        nodes = self.rstg(nodes)
        # nodes: B x T x C x N
        if self.project_i3d and self.params.project_at_input and self.params.remap_first==False:
            nodes = nodes.view(self.BT, self.C, self.num_nodes)
            nodes = self.project_back_i3d_linear(nodes)
            nodes = nodes.view(self.B, self.T, self.backbone_dim, self.num_nodes)

        out = self.remap_nodes(nodes)

        # if self.params.isolate_graphs:
        #     return input_feats

        if self.project_i3d and self.params.project_at_input and self.params.remap_first==True:
            out = out.view(self.B*self.T, self.C, self.H, self.W)
            out = self.project_back_i3d_linear(out)
        else:
            out = out.view(self.B*self.T, self.backbone_dim, self.H, self.W)
        # out: B x T x C x H x W
        return out


class DynamicGraphWrapper(nn.Module):
    def __init__(self, block, 
            H=14, W=14, oH=7, oW=7, iH=None, iW=None,
            h=3, w=3,
            node_dim=512, in_channels=1024, out_num_ch = 2048,
            project_i3d=False, name=''):
        super(DynamicGraphWrapper, self).__init__()
        self.block = block
        self.dynamic_graph = DynamicGraph(backbone_dim=in_channels,
            H=H,W=W,oH=oH, oW=oW, iH=iH, iW=iW,
            h=h,w=w,
            node_dim=node_dim, out_num_ch =out_num_ch,
            project_i3d=project_i3d, name=name)
        self.in_channels = in_channels
        self.H = H
        self.W = W
        # the dimenstion of the input used at training time
        # to be used for resize for fullsize evaluation
        self.iH = iH
        self.iW = iW
        # self.n_segment = n_segment

        if args.graph_residual_type == 'norm':
            # residual branch
            # initialize the residual branch with 0, 
            # so that it is ignored at the begening of the optimization
            self.norm_dict = nn.ModuleDict({})
            # residual norm
            zero_init = True
            if args.rstg_combine == 'serial':
                zero_init = False
            self.norm_dict['residual_norm'] = LayerNormAffine2D(self.in_channels, (self.in_channels, self.H, self.W),
                zero_init=zero_init)
        elif args.graph_residual_type == 'out_gate':
            self.out_gate = nn.Sequential(
                                    nn.Conv2d(in_channels, in_channels, [1, 1]),
                                    nn.Sigmoid()
                                )
            # init bias from conv layer
            nn.init.constant_(self.out_gate[0].bias, -1.0)
        elif args.graph_residual_type == 'gru_gate':
            self.gru_gate_r_w = nn.Conv2d(in_channels, in_channels, [1, 1])
            self.gru_gate_r_u = nn.Conv2d(in_channels, in_channels, [1, 1])
            self.gru_gate_z_w = nn.Conv2d(in_channels, in_channels, [1, 1])
            self.gru_gate_z_u = nn.Conv2d(in_channels, in_channels, [1, 1])
            self.gru_gate_z_b = torch.nn.Parameter(torch.Tensor(size=[1,in_channels,1,1]))
            self.gru_gate_h_w = nn.Conv2d(in_channels, in_channels, [1, 1])
            self.gru_gate_h_u = nn.Conv2d(in_channels, in_channels, [1, 1])
            nn.init.constant_(self.gru_gate_z_b, -2.0)
        elif args.graph_residual_type == '1chan_out_gate':
            self.one_chan_out_gate = nn.Sequential(
                                            nn.Conv2d(in_channels, 1, [1, 1]),
                                            nn.Sigmoid()
                                        )
            # init bias from conv layer
            nn.init.constant_(self.one_chan_out_gate[0].bias, -1.0)
        # for full resolution testing, we resize the input to the graph
        if args.full_res == 'resize_graph_input':
            self.input_pooling = nn.AdaptiveAvgPool2d((self.H,self.W))
            self.output_pooling = nn.AdaptiveAvgPool2d((self.H // 7 * 8,self.W // 7 * 8))

    def get_norm(self, input, name):
        # input: B*T x C x H x W
        return self.norm_dict[name](input)

    def get_gru_gate(self, x, y):
        # GRU-like residual gate
        r = nn.Sigmoid()(self.gru_gate_r_w(y) + self.gru_gate_r_u(x))
        z = nn.Sigmoid()(self.gru_gate_z_w(y) + self.gru_gate_z_u(x)) + self.gru_gate_z_b
        h = nn.Tanh()(self.gru_gate_h_w(y) + self.gru_gate_h_u(r*x))
        return  (1-z)*x + z*h

    def forward(self, inp):
        inp = self.block(inp)
        # if args.freeze_regions == 'offsets_input':
        #     inp = inp.detach()

        # print(f'block {x.shape}')

        if 'resnet50_smt_else' in  args.arch:
            x = inp.permute([0,2,1,3,4]).contiguous()
            x = x.view(x.shape[0]*x.shape[1], x.shape[2], x.shape[3], x.shape[4])
        else:
            x = inp

        if args.full_res == 'resize_graph_input':
            x = self.input_pooling(x)

        y = self.dynamic_graph(x)

        if args.graph_residual_type == 'norm':
            y = self.get_norm(y, 'residual_norm')
        elif args.graph_residual_type == 'out_gate':
            y = self.out_gate(y)  * y
        elif args.graph_residual_type == 'gru_gate':
            y = self.get_gru_gate(x,y)
        elif args.graph_residual_type == '1chan_out_gate':
            y = self.one_chan_out_gate(y)  * y

        if args.full_res == 'resize_graph_input':
            y = self.output_pooling(y)

        if args.isolate_graphs:
            out = inp[0]
        else:
            if args.rstg_combine == 'serial':
                out = y
            elif args.rstg_combine == 'plus':
                if type(inp) is tuple:
                    out = inp[0] + y
                else:
                    out = inp + y

        if 'resnet50_smt_else' in  args.arch:
            out = out.view((inp.shape[0], inp.shape[2], out.shape[1], out.shape[2], out.shape[3]))
            out = out.permute([0,2,1,3,4]).contiguous()
        
        return out


def make_rstg(net):
    import torchvision
 
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

    if True:
    # if isinstance(net, torchvision.models.ResNet):
        
        for place in places:
            layer = int(place.split('.')[0])
            block = int(place.split('.')[1])
            if args.bottleneck_graph:
                
                # list_layer[layer][block].conv2 = nn.Sequential(list_layer[layer][block].conv2, DynamicGraphWrapper(list_layer[layer][block].conv2,
                list_layer[layer][block].conv2 = DynamicGraphWrapper(list_layer[layer][block].conv2, 

                    in_channels=graph_params[layer]['in_channels'], 
                    H=graph_params[layer]['H'], W=graph_params[layer]['H'], 
                    oH=out_pool_size, oW=out_pool_size, 
                    iH=graph_params[layer]['iH'], iW=graph_params[layer]['iH'],
                    out_num_ch = out_num_ch,
                    node_dim=graph_params[layer]['node_dim'], project_i3d=graph_params[layer]['project_i3d'],
                    name=graph_params[layer]['name'])
                #)
            else:
                list_layer[layer][block] = DynamicGraphWrapper(list_layer[layer][block], 
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
    else:
        raise NotImplementedError


def save_image(out, name='img'):
    out = out.permute(0,2,3,4,1)
    out_image = out.detach().numpy()[0,0]
    out_image = out_image - out_image.min()
    out_image = out_image / out_image.max()
    f, axarr = plt.subplots(1,1)
    axarr.imshow(out_image) 
    plt.savefig(name+'.jpeg') 

