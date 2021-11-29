
# Code for RSTG model
# Recurrent Space-time Graph Neural Networks - RSTG
# adapted from https://github.com/IuliaDuta/RSTG

import torch
from torch import nn
from torch.nn import functional as F
import sys
import os
sys.path.append(os.path.abspath(os.path.join(
    os.path.dirname(os.path.realpath(__file__)), '..')))
from opts import parser
from ops.models_utils import *

global args, best_prec1
from opts import parse_args
args = parser.parse_args()
args = parse_args()
import pdb


class RSTG(nn.Module):
    def __init__(self,params, backbone_dim=1024, node_dim=512, project_i3d=True):

        super(RSTG, self).__init__()
        self.params = params
        self.backbone_dim = backbone_dim
        self.node_dim = node_dim
        self.number_iterations = 3
        self.num_nodes = 9
        self.project_i3d = project_i3d
        self.norm_dict = nn.ModuleDict({})

        # intern LSTM
        self.internal_lstm = recurrent_net(batch_first=True,
             input_size=self.node_dim, hidden_size=self.node_dim)
        # extern LSTM
        self.external_lstm = recurrent_net(batch_first=True,
             input_size=self.node_dim, hidden_size=self.node_dim)

        # send function
        if self.params.send_layers == 1:
                self.send_mlp = nn.Sequential(
                    torch.nn.Linear(self.node_dim, self.node_dim),
                    torch.nn.ReLU()
                )
        elif self.params.send_layers == 2:
            if self.params.combine_by_sum:
                comb_mult = 1
            else:
                comb_mult = 2
            self.send_mlp = nn.Sequential(
                torch.nn.Linear(comb_mult * self.node_dim, self.node_dim),
                torch.nn.ReLU(),
                torch.nn.Linear(self.node_dim, self.node_dim),
                torch.nn.ReLU()
            )

        # norm send 
        self.norm_dict['send_message_norm'] = LayerNormAffineXC(
            self.node_dim, (self.num_nodes,self.num_nodes,self.node_dim)
        )
        self.norm_dict['update_norm'] = LayerNormAffineXC(
            self.node_dim, (self.num_nodes, self.node_dim)
        )
        self.norm_dict['before_graph_norm'] = LayerNormAffineXC(
            self.node_dim, (self.num_nodes, self.node_dim)
        )


        
        # attention function
        if 'dot' in self.params.aggregation_type:
            self.att_q = torch.nn.Linear(self.node_dim, self.node_dim)
            self.att_k = torch.nn.Linear(self.node_dim, self.node_dim)
            # attention bias
            self.att_bias = torch.nn.Parameter(
                torch.zeros(size=[1,1,1,self.node_dim], requires_grad=True)
            )
        
        # update function
        if self.params.update_layers == 0:
            self.update_mlp = nn.Identity()
        elif self.params.update_layers == 1:
            self.update_mlp = nn.Sequential(
                torch.nn.Linear(self.node_dim, self.node_dim),
                torch.nn.ReLU()
            )
        elif self.params.update_layers == 2:
            if self.params.combine_by_sum:
                comb_mult = 1
            else:
                comb_mult = 2
            self.update_mlp = nn.Sequential(
                torch.nn.Linear(comb_mult * self.node_dim, self.node_dim),
                torch.nn.ReLU(),
                torch.nn.Linear(self.node_dim, self.node_dim),
                torch.nn.ReLU()
            )

    def get_norm(self, input, name, zero_init=False):
        # input: B x N x T x C 
        #     or B x N x N x T x C
        if len(input.size()) == 5:
            input = input.permute(0,3,1,2,4).contiguous()
        elif len(input.size()) == 4:
            input = input.permute(0,2,1,3).contiguous()
      
        norm = self.norm_dict[name]

        input = norm(input)
        if len(input.size()) == 5:
            input = input.permute(0,2,3,1,4).contiguous()
        elif len(input.size()) == 4:
            input = input.permute(0,2,1,3).contiguous()
        return input

    def send_messages(self, nodes):
        # nodes: B x N x T x C
        # nodes1: B x 1 x N x T x C
        # nodes2: B x N x 1 x T x C
        nodes1 = nodes.unsqueeze(1)
        nodes2 = nodes.unsqueeze(2)

        nodes1 = nodes1.repeat(1,self.num_nodes,1,1,1)
        nodes2 = nodes2.repeat(1,1,self.num_nodes,1,1)
        # B x N x N x T x 2C
        if self.params.combine_by_sum:
            messages = nodes1 + nodes2
        else:
            messages = torch.cat((nodes1, nodes2),dim=-1)
        messages = self.send_mlp(messages)
        messages = self.get_norm(messages, 'send_message_norm')
        return messages
    
    def aggregate(self, nodes, messages):
        if 'sum' in self.params.aggregation_type:
            return self.sum_aggregate(messages)
        elif 'dot' in self.params.aggregation_type:
            return self.dot_attention(nodes, messages)
    
    def sum_aggregate(self, messages):
        # nodes:    B x N x T x C
        # messages: B x NxN x T x C
        return messages.mean(dim=2)

    def dot_attention(self, nodes, messages):
        # nodes:    B x N x T x C
        # messages: B x NxN x T x C

        # nodes1: B x N x 1 x T x C
        # nodes2: B x 1 x N x T x C
        # corr B x N x N x T
        
        nodes_q = self.att_q(nodes)
        nodes_k = self.att_k(nodes)

        nodes_q = nodes_q.permute(0,2,1,3)
        nodes_k = nodes_k.permute(0,2,3,1)

        corr = torch.matmul(nodes_q, nodes_k).unsqueeze(-1)
        corr = corr.permute(0,2,3,1,4)

        nodes = F.softmax(corr, dim=2) *  messages
        nodes = nodes.sum(dim=2)

        nodes = nodes + self.att_bias
        nodes = F.relu(nodes)
        return nodes

    def update_nodes(self, nodes, aggregated):
        if self.params.combine_by_sum:
            upd_input = nodes + aggregated
        else:
            upd_input = torch.cat((nodes, aggregated), dim=-1)
        nodes = self.update_mlp(upd_input)
        nodes = self.get_norm(nodes, 'update_norm')
        return nodes

    def forward(self, input):
        self.B = input.shape[0]
        self.T = input.shape[1]

        # input RSTG: B x T x C x H x W
        # set input: ... -> B x T x C x N

        # for LSTM we need (B * N) x T x C
        # propagation: B x 1 x N x T x C 
                    #   + B x N x 1 x T x C
                    #    (B x N*N x T) x C => liniar
        # nodes: B x N x T x C
        nodes = input.permute(0,3,1,2)
        
        nodes = self.get_norm(nodes, 'before_graph_norm')
        time_iter_mom = [0, 1, 2]
        
        for space_iter in range(self.number_iterations):
            # internal time processing
            if space_iter in time_iter_mom:
                nodes = nodes.view(self.B * self.num_nodes, self.T, self.node_dim)
                self.internal_lstm.flatten_parameters()
                nodes, _ = self.internal_lstm(nodes)
                nodes = nodes.view(self.B, self.num_nodes, self.T, self.node_dim)
            
            # space_processing: send, aggregate, update
            messages        = self.send_messages(nodes)
            aggregated      = self.aggregate(nodes, messages)
            nodes           = self.update_nodes(nodes, aggregated)
            
        # external time processing
        nodes = nodes.view(self.B * self.num_nodes, self.T, self.node_dim)
        self.external_lstm.flatten_parameters()
        nodes, _ = self.external_lstm(nodes)
        nodes = nodes.view(self.B, self.num_nodes, self.T, self.node_dim)
        

        # B x N x T x C -> B x T x C x N
        nodes = nodes.permute(0,2,3,1).contiguous()
        
        return nodes
