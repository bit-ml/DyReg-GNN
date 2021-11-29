# Code adapted from "TSM: Temporal Shift Module for Efficient Video Understanding"
# Ji Lin*, Chuang Gan, Song Han

from torch import nn

from ops.basic_ops import ConsensusModule
from ops.resnet3d_xl import Net
from ops.resnet2d import wide_resnet50_2
from ops.resnet2d import resnet101
from ops.resnet2d import resnet50
from ops.resnet2d import resnet34
from ops.resnet2d import resnet18


from ops.transforms import *
from torch.nn.init import normal_, constant_
import pdb
global args
from opts import parse_args
args = parse_args()
from ops.dyreg import *


class TSN(nn.Module):
    def __init__(self, num_class, num_segments, modality,
                 base_model='resnet101', new_length=None,
                 consensus_type='avg', before_softmax=True,
                 dropout=0.8, img_feature_dim=256,
                 crop_num=1, partial_bn=True, print_spec=True, pretrain='imagenet',
                 is_shift=False, shift_div=8, shift_place='blockres', fc_lr5=False,
                 temporal_pool=False, non_local=False):
        super(TSN, self).__init__()
        self.modality = modality
        self.num_segments = num_segments
        self.reshape = True
        self.before_softmax = before_softmax
        self.dropout = dropout
        self.crop_num = crop_num
        self.consensus_type = consensus_type
        self.img_feature_dim = img_feature_dim  # the dimension of the CNN feature to represent each frame
        self.pretrain = pretrain

        self.is_shift = is_shift
        self.shift_div = shift_div
        self.shift_place = shift_place
        self.base_model_name = base_model
        self.fc_lr5 = fc_lr5
        self.temporal_pool = temporal_pool
        self.non_local = non_local
        if args.arch == 'resnet13':
            self.out_pool_size = 8
            self.out_num_ch = 256
        elif args.arch == 'resnet18':
            self.out_pool_size = 7
            self.out_num_ch = 512
        elif args.arch == 'resnet34':
            self.out_pool_size = 7
            self.out_num_ch = 512
        elif 'resnet50' in args.arch:
            self.out_pool_size = 7
            self.out_num_ch = 2048
            if args.full_res:
                self.out_pool_size = 8
        elif args.arch == 'wide_resnet50_2':
            self.out_pool_size = 7
            self.out_num_ch = 512
        elif 'resnet101' in args.arch:
            self.out_pool_size = 7
            self.out_num_ch = 2048
            if args.full_res:
                self.out_pool_size = 8

        # auxiliary intermediate features
        self.interm_feats = {}
        self.kernels = {}
        if not before_softmax and consensus_type != 'avg':
            raise ValueError("Only avg consensus can be used after Softmax")

        if new_length is None:
            self.new_length = 1 if modality == "RGB" or modality == 'gray' else 5
        else:
            self.new_length = new_length
        
        places = args.place_graph.replace('layer','').split('_')
        self.places = []
        for place in places:
            layer = int(place.split('.')[0])
            block = int(place.split('.')[1])
            self.places.append((layer, block))
        
        
        if print_spec:
            print(("""
    Initializing TSN with base model: {}.
    TSN Configurations:
        input_modality:     {}
        num_segments:       {}
        new_length:         {}
        consensus_module:   {}
        dropout_ratio:      {}
        img_feature_dim:    {}
            """.format(base_model, self.modality, self.num_segments, self.new_length, consensus_type, self.dropout, self.img_feature_dim)))

        self._prepare_base_model(base_model)

        feature_dim = self._prepare_tsn(num_class)

        self.consensus = ConsensusModule(consensus_type)

        if not self.before_softmax:
            self.softmax = nn.Softmax()

        self._enable_pbn = partial_bn
        if partial_bn:
            self.partialBN(True)

    def _prepare_tsn(self, num_class):
        feature_dim = getattr(self.base_model, self.base_model.last_layer_name).in_features
        if self.dropout == 0:
            setattr(self.base_model, self.base_model.last_layer_name, nn.Linear(feature_dim, num_class))
            self.new_fc = None
        else:
            setattr(self.base_model, self.base_model.last_layer_name, nn.Dropout(p=self.dropout))
            self.new_fc = nn.Linear(self.out_num_ch, num_class)

        std = 0.001
        if self.new_fc is None:
            normal_(getattr(self.base_model, self.base_model.last_layer_name).weight, 0, std)
            constant_(getattr(self.base_model, self.base_model.last_layer_name).bias, 0)
        else:
            if hasattr(self.new_fc, 'weight'):
                normal_(self.new_fc.weight, 0, std)
                constant_(self.new_fc.bias, 0)
        return feature_dim

    def _prepare_base_model(self, base_model):
        print('=> base model: {}'.format(base_model))

        if 'resnet' in base_model:
            
            if base_model == 'resnet13':
                # do NOT load pretrained imagenet
                self.base_model = resnet18(pretrained=False)
            elif base_model == 'resnet18':
                self.base_model = resnet18(pretrained=False if args.resume else True)
            elif base_model == 'resnet34':
                self.base_model = resnet34(pretrained=False if args.resume else True)
            elif base_model == 'resnet50':
                self.base_model = resnet50(pretrained=False if args.resume else True )
            elif base_model == 'resnet101':
                self.base_model = resnet101(pretrained=False if args.resume else True )
            elif base_model == 'wide_resnet50_2':
                self.base_model = wide_resnet50_2(pretrained=False if args.resume else True )


            if args.dataset =='syncMNIST' or args.dataset == 'multiSyncMNIST':
                # self.base_model.layer3 = self.base_model.layer4
                self.base_model.layer4 = nn.Sequential(nn.Identity())
                self.base_model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

            if self.is_shift:
                print('Adding temporal shift...')
                from ops.temporal_shift import make_temporal_shift
                make_temporal_shift(self.base_model, self.num_segments,
                                    n_div=self.shift_div, place=self.shift_place, temporal_pool=self.temporal_pool)
            if self.non_local:
                print('Adding non-local module...')
                from ops.non_local import make_non_local
                make_non_local(self.base_model, self.num_segments)
            if args.use_rstg:
                print('Adding rstg module...')
                from ops.dyreg import make_rstg
                make_rstg(self.base_model)

            if args.dataset =='somethingv2' or args.dataset =='something' or args.dataset =='cater' or args.dataset =='others':
                if 'resnet50_smt_else' in base_model:
                    self.base_model.last_layer_name = 'classifier'
                else:
                    self.base_model.last_layer_name = 'fc'
                self.input_size = 224
                self.input_mean = [0.485, 0.456, 0.406]
                self.input_std = [0.229, 0.224, 0.225]

                # self.base_model.avgpool = nn.AdaptiveAvgPool2d(1)
                self.base_model.avgpool = nn.Identity()
                self.base_model.rstg_avgpool = nn.AdaptiveAvgPool2d(1)

                self.norm_dict = nn.ModuleDict({})

                if args.rstg_skip_connection == True:
                    self.map_final_project = nn.Conv2d(self.out_num_ch//args.ch_div, self.out_num_ch, [1, 1])
                    if args.tmp_norm_skip_conn:
                        if args.freeze_backbone or args.init_skip_zero:
                            zero_init = True
                        else:
                            zero_init = False
                        self.norm_dict['skip_conn'] = LayerNormAffine2D(self.out_num_ch, (self.out_num_ch, 7, 7),
                                                                zero_init=zero_init)

            
            elif args.dataset =='syncMNIST' or args.dataset == 'multiSyncMNIST':
                self.base_model.last_layer_name = 'fc'
                self.input_size = 128
                self.input_mean = [0]
                self.input_std = [1]

                self.base_model.avgpool = nn.Identity()
                self.base_model.rstg_avgpool = nn.AdaptiveAvgPool2d(1)
                self.norm_dict = nn.ModuleDict({})

                if args.rstg_skip_connection == True:
                    self.map_final_project = nn.Conv2d(self.out_num_ch//args.ch_div, self.out_num_ch, [1, 1])
                    if args.tmp_norm_skip_conn:
                        if args.freeze_backbone:
                            zero_init = True
                        else:
                            zero_init = False
                        self.norm_dict['skip_conn'] = LayerNormAffine2D(self.out_num_ch, (self.out_num_ch, 7, 7),
                                                                zero_init=zero_init)


    def train(self, mode=True):
        """
        Override the default train() to freeze the BN parameters
        :return:
        """
        super(TSN, self).train(mode)
        count = 0
        if self._enable_pbn and mode:
            print("Freezing BatchNorm2D")
            for m in self.base_model.modules():
                if isinstance(m, nn.BatchNorm2d):
                    # count += 1
                    # if count >= (2 if self._enable_pbn else 1):
                    m.eval()
                    # shutdown update in frozen mode
                    m.weight.requires_grad = False
                    m.bias.requires_grad = False


    def partialBN(self, enable):
        self._enable_pbn = enable
        

    def get_optim_policies(self):
        first_conv_weight = []
        first_conv_bias = []
        normal_weight = []
        normal_bias = []
        lr5_weight = []
        lr10_bias = []
        bn = []
        custom_ops = []

        conv_cnt = 0
        bn_cnt = 0

        for (m_name, m) in self.named_modules():

            if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.Conv1d) or isinstance(m, torch.nn.Conv3d):
                ps = list(m.named_parameters())
                ps = [(m_name+'.'+x[0], x[1]) for x in ps]
                if ps[0][1].requires_grad == False:
                    continue

                conv_cnt += 1
                if conv_cnt == 1:
                    first_conv_weight.append(ps[0])
                    if len(ps) == 2:
                        first_conv_bias.append(ps[1])
                else:
                    normal_weight.append(ps[0])
                    if len(ps) == 2:
                        normal_bias.append(ps[1])
            elif isinstance(m, torch.nn.Linear):
                # TODO: ce inseamna hardcodarile astea 56, 56?? 
                if 174 in m.weight.shape or 46 in m.weight.shape or 56 in m.weight.shape:
                    print(f'Last layer: Prediction: {m}')
                    ps = list(m.named_parameters())
                    ps = [(m_name+'.'+x[0], x[1]) for x in ps]
                    if ps[0][1].requires_grad == False:
                        continue
                    if self.fc_lr5:
                        lr5_weight.append(ps[0])
                    else:
                        normal_weight.append(ps[0])
                    
                    if len(ps) == 2:
                        if self.fc_lr5:
                            lr10_bias.append(ps[1])
                        else:
                            normal_bias.append(ps[1])
                else:
                    ps = list(m.named_parameters())
                    ps = [(m_name+'.'+x[0], x[1]) for x in ps]
                    if ps[0][1].requires_grad == False:
                        continue
                    normal_weight.append(ps[0])
                    normal_bias.append(ps[1])

            elif isinstance(m, torch.nn.BatchNorm2d):
                bn_cnt += 1
                # later BN's are frozen
                if not self._enable_pbn or bn_cnt == 1:
                    ps = list(m.named_parameters())
                    ps = [(m_name+'.'+x[0], x[1]) for x in ps]
                    if ps[0][1].requires_grad == False:
                        continue
                    bn.extend(ps)
            elif isinstance(m, torch.nn.SyncBatchNorm):
                bn_cnt += 1
                # later BN's are frozen
                if not self._enable_pbn or bn_cnt == 1:
                    ps = list(m.named_parameters())
                    ps = [(m_name+'.'+x[0], x[1]) for x in ps]
                    if ps[0][1].requires_grad == False:
                        continue
                    bn.extend(ps)
            elif isinstance(m, torch.nn.BatchNorm3d):
                bn_cnt += 1
                # later BN's are frozen
                if not self._enable_pbn or bn_cnt == 1:
                    ps = list(m.named_parameters())
                    ps = [(m_name+'.'+x[0], x[1]) for x in ps]
                    if ps[0][1].requires_grad == False:
                        continue    
                    bn.extend(ps)
            elif isinstance(m, torch.nn.GRU) or isinstance(m, torch.nn.LSTM):
                ps = list(m.named_parameters())
                ps = [(m_name+'.'+x[0], x[1]) for x in ps]
                if ps[0][1].requires_grad == False:
                    continue
                normal_weight.append(ps[0])
                normal_weight.append(ps[1])

                normal_bias.append(ps[2])
                normal_bias.append(ps[3])
            elif (isinstance(m, LayerNormAffineXC)
                    or isinstance(m, LayerNormAffine2D) or isinstance(m, LayerNormAffine1D) ) :
                ps = list(m.named_parameters())
                ps = [(m_name+'.'+x[0], x[1]) for x in ps]
                if ps[0][1].requires_grad == False:
                    continue
                # named_parameter = list(m.named_parameters())
                bn.extend(ps)
            elif isinstance(m, torch.nn.ConvTranspose2d):
                ps = list(m.named_parameters())
                ps = [(m_name+'.'+x[0], x[1]) for x in ps]
                if ps[0][1].requires_grad == False:
                    continue
                normal_weight.append(ps[0])
                normal_bias.append(ps[1])
            elif isinstance(m, torch.nn.Parameter):
                ps = list(m.named_parameters())
                if ps[0][1].requires_grad == False:
                    continue
                print(m)
            elif len(m._modules) == 0:
                if len(list(m.named_parameters())) > 0:
                    raise ValueError("New atomic module type: {}. Need to give it a learning policy".format(type(m))) 
    
        additional_params_name = ['offset_alpha', 'att_bias']
        for name, param in self.named_parameters():
            for add_p in additional_params_name:
                if add_p in name:
                    if param.requires_grad == False:
                        continue
                    normal_weight.append( (name, param))
                
        lr_policies =  [
            {'params': first_conv_weight, 'lr_mult': 5 if self.modality == 'Flow' else 1, 'decay_mult': 1,
             'name': "first_conv_weight"},
            {'params': first_conv_bias, 'lr_mult': 10 if self.modality == 'Flow' else 2, 'decay_mult': 0,
             'name': "first_conv_bias"},
            {'params': normal_weight, 'lr_mult': 1, 'decay_mult': 1,
             'name': "normal_weight"},
            {'params': normal_bias, 'lr_mult': 2, 'decay_mult': 0,
             'name': "normal_bias"},
            {'params': bn, 'lr_mult': 1, 'decay_mult': 0,
             'name': "BN scale/shift"},
            {'params': custom_ops, 'lr_mult': 1, 'decay_mult': 1,
             'name': "custom_ops"},
            # for fc
            {'params': lr5_weight, 'lr_mult': 5, 'decay_mult': 1,
             'name': "lr5_weight"},
            {'params': lr10_bias, 'lr_mult': 10, 'decay_mult': 0,
             'name': "lr10_bias"},
        ]

        if args.dataset =='syncMNIST' or args.dataset == 'multiSyncMNIST':
            # set all lr mult, decay mult to 1 for all layers
            for p in lr_policies:
                p['lr_mult'] = 1
                p['decay_mult'] = 1
            
        return lr_policies



    def forward(self, input, no_reshape=False):
        if not no_reshape:
            sample_len = (3 if self.modality == "RGB" else 2) * self.new_length
            if self.modality == 'gray':
                sample_len = 1 * self.new_length
            base_out = self.base_model(input.view((-1, sample_len) + input.size()[-2:]))
        else:
            
            base_out = self.base_model(input)
            # B x C x T x H x W



        if args.use_rstg:
            for layer, block in self.places:
                if args.bottleneck_graph:
                    if layer == 1:
                        dynamic_graph1 = self.base_model.layer1[block].conv2[1].dynamic_graph
                    elif layer == 2:
                        dynamic_graph2 = self.base_model.layer2[block].conv2[1].dynamic_graph
                    elif layer == 3:
                        dynamic_graph3 = self.base_model.layer3[block].conv2[1].dynamic_graph
                    elif layer == 4:
                        dynamic_graph4 = self.base_model.layer4[block].conv2[1].dynamic_graph
                else:
                    if layer == 1:
                        dynamic_graph1 = self.base_model.layer1[block].dynamic_graph
                    elif layer == 2:
                        dynamic_graph2 = self.base_model.layer2[block].dynamic_graph
                    elif layer == 3:
                        dynamic_graph3 = self.base_model.layer3[block].dynamic_graph
                    elif layer == 4:
                        dynamic_graph4 = self.base_model.layer4[block].dynamic_graph

            for layer, block in self.places:
                
                if layer == 1:
                    self.interm_feats[f'layer{layer}_{block}_kernels'] = dynamic_graph1.kernel
                    self.kernels[f'layer{layer}_block{block}'] = dynamic_graph1.kernel
                    
                elif layer == 2:
                    self.interm_feats[f'layer{layer}_{block}_kernels'] = dynamic_graph2.kernel
                    self.kernels[f'layer{layer}_block{block}'] = dynamic_graph2.kernel

                elif layer == 3:
                    self.interm_feats[f'layer{layer}_{block}_kernels'] =dynamic_graph3.kernel
                    self.kernels[f'layer{layer}_block{block}'] = dynamic_graph3.kernel

                elif layer == 4:
                    self.interm_feats[f'layer{layer}_{block}_kernels'] = dynamic_graph4.kernel
                    self.kernels[f'layer{layer}_block{block}'] = dynamic_graph4.kernel

                 

        else:
            self.interm_feats['layer3_kernels'] = None

        base_out = base_out.view((base_out.shape[0],self.out_num_ch, self.out_pool_size, self.out_pool_size))
        self.offsets = {}
        self.offsets_normed = {}
        self.nodes = {}
        if args.use_rstg and args.rstg_skip_connection:
            all_graphs_sum = 0
            for layer, block in self.places:
                if layer == 1:
                    all_graphs_sum = all_graphs_sum + dynamic_graph1.graph_out_map
                elif layer == 2:
                    all_graphs_sum = all_graphs_sum + dynamic_graph2.graph_out_map
                elif layer == 3:
                    all_graphs_sum = all_graphs_sum + dynamic_graph3.graph_out_map
                elif layer == 4:
                    all_graphs_sum = all_graphs_sum + dynamic_graph4.graph_out_map


            all_graphs_sum = self.map_final_project(all_graphs_sum)
            base_out = base_out + all_graphs_sum

        self.aux_feats = {}
        if args.use_rstg:
            for layer, block in self.places:
                if layer == 1:
                    self.offsets[f'layer1_block{block}'] = dynamic_graph1.offsets
                    self.offsets_normed[f'layer1_block{block}'] = dynamic_graph1.offsets / args.graph_params[layer]['H']
                    self.nodes[f'layer1_block{block}'] = dynamic_graph1.nodes_wo_position
                    for key in dynamic_graph1.aux_feats.keys():
                        if key not in self.aux_feats:
                            self.aux_feats[key] = {}
                        self.aux_feats[key][f'layer1_block{block}'] = dynamic_graph1.aux_feats[key]
                elif layer == 2:
                    self.offsets[f'layer2_block{block}'] = dynamic_graph2.offsets
                    self.offsets_normed[f'layer2_block{block}'] = dynamic_graph2.offsets / args.graph_params[layer]['H']
                    self.nodes[f'layer2_block{block}'] = dynamic_graph2.nodes_wo_position
                    for key in dynamic_graph2.aux_feats.keys():
                        if key not in self.aux_feats:
                            self.aux_feats[key] = {}
                        self.aux_feats[key][f'layer2_block{block}'] = dynamic_graph2.aux_feats[key]
                elif layer == 3:
                    self.offsets[f'layer3_block{block}'] = dynamic_graph3.offsets
                    self.offsets_normed[f'layer3_block{block}'] = dynamic_graph3.offsets / args.graph_params[layer]['H']
                    self.nodes[f'layer3_block{block}'] = dynamic_graph3.nodes_wo_position
                    for key in dynamic_graph3.aux_feats.keys():
                        if key not in self.aux_feats:
                            self.aux_feats[key] = {}
                        self.aux_feats[key][f'layer3_block{block}'] = dynamic_graph3.aux_feats[key]
                elif layer == 4:
                    self.offsets[f'layer4_block{block}'] = dynamic_graph4.offsets
                    self.offsets_normed[f'layer4_block{block}'] = dynamic_graph4.offsets / args.graph_params[layer]['H']
                    self.nodes[f'layer4_block{block}'] = dynamic_graph4.nodes_wo_position
                    for key in dynamic_graph4.aux_feats.keys():
                        if key not in self.aux_feats:
                            self.aux_feats[key] = {}
                        self.aux_feats[key][f'layer4_block{block}'] = dynamic_graph4.aux_feats[key]

        base_out = self.base_model.rstg_avgpool(base_out)
        base_out = torch.flatten(base_out,1)

        if self.dropout > 0:
            base_out = self.new_fc(base_out)

        if not self.before_softmax:
            base_out = self.softmax(base_out)

        if self.reshape:
            if (self.is_shift and self.temporal_pool):
                base_out = base_out.view((-1, self.num_segments // 2) + base_out.size()[1:])
            else:
                base_out = base_out.view((-1, args.num_segments) + base_out.size()[1:])

            output = self.consensus(base_out)

        model_aux_feats = { 
            'offsets' : self.offsets,
            'offsets_normed' : self.offsets_normed,
            'nodes' :self.nodes,
            'interm_feats' : self.interm_feats,
            'kernel': self.kernels
        }
        model_aux_feats.update(self.aux_feats)
        return output.squeeze(1), model_aux_feats

    @property
    def crop_size(self):
        return self.input_size

    @property
    def scale_size(self):
        return self.input_size * 256 // 224

    def get_augmentation(self, flip=True):
        
        if flip:
            return torchvision.transforms.Compose([GroupMultiScaleCrop(self.input_size, [1, .875, .75, .66]),
                                                    GroupRandomHorizontalFlip(is_flow=False)])
        else:
            print('#' * 20, 'NO FLIP!!!')
            return torchvision.transforms.Compose([GroupMultiScaleCrop(self.input_size, [1, .875, .75, .66])])
            