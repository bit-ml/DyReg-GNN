# Code for "TSM: Temporal Shift Module for Efficient Video Understanding"
# arXiv:1811.08383
# Ji Lin*, Chuang Gan, Song Han
# {jilin, songhan}@mit.edu, ganchuang@csail.mit.edu
import argparse
def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


parser = argparse.ArgumentParser(description="PyTorch implementation of Temporal Segment Networks")
parser.add_argument('--dataset', type=str, default='somethingv2')
parser.add_argument('--dataset_fraction_used', default=1.0, type=float,   
     help='use just x% of the total dataset')

parser.add_argument('--test_dataset', type=str, default='../test')
parser.add_argument('--train_dataset', type=str, default='../train')
parser.add_argument('--modality', type=str, choices=['RGB', 'Flow', 'gray'], default='RGB')
parser.add_argument('--train_list', type=str, default="")
parser.add_argument('--val_list', type=str, default="")
parser.add_argument('--root_path', type=str, default="")
parser.add_argument('--store_name', type=str, default="")
# ========================= Model Configs ==========================
parser.add_argument('--arch', type=str, default="resnet50")
parser.add_argument('--num_segments', type=int, default=16)

parser.add_argument('--consensus_type', type=str, default='avg')
parser.add_argument('--k', type=int, default=3)

parser.add_argument('--dropout', '--do', default=0.5, type=float,
                    metavar='DO', help='dropout ratio (default: 0.5)')
parser.add_argument('--loss_type', type=str, default="nll",
                    choices=['nll'])
parser.add_argument('--img_feature_dim', default=256, type=int, help="the feature dimension for each frame")
parser.add_argument('--suffix', type=str, default=None)
parser.add_argument('--pretrain', type=str, default='imagenet')
parser.add_argument('--tune_from', type=str, default=None, help='fine-tune from checkpoint')

# ========================= Learning Configs ==========================
parser.add_argument('--epochs', default=120, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('-b', '--batch-size', default=10, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=0.001, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--lr_type', default='step', type=str,
                    metavar='LRtype', help='learning rate type')
parser.add_argument('--lr_steps', default=[50, 100], type=float, nargs="+",
                    metavar='LRSteps', help='epochs to decay learning rate by 10')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float,
                    metavar='W', help='weight decay (default: 5e-4)')
parser.add_argument('--clip-gradient', '--gd', default=None, type=float,
                    metavar='W', help='gradient norm clipping (default: disabled)')
parser.add_argument('--no_partialbn', '--npb', default=False, action="store_true")

# no_partialbn == NOT FREEZE
# partialbn == enable_partialbn == FREZE BN (except the first one)

# ========================= Monitor Configs ==========================
parser.add_argument('--print-freq', '-p', default=20, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--eval-freq', '-ef', default=5, type=int,
                    metavar='N', help='evaluation frequency (default: 5)')


# ========================= Runtime Configs ==========================
parser.add_argument('-j', '--workers', default=5, type=int, metavar='N',
                    help='number of data loading workers (default: 8)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--replace_ignore', default=True, type=str2bool,
                    help='change name in resnet backbone')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--snapshot_pref', type=str, default="")
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--gpus', nargs='+', type=int, default=None)
parser.add_argument('--flow_prefix', default="", type=str)
parser.add_argument('--root_log',type=str, default='log')
parser.add_argument('--root_model', type=str, default='checkpoint')
parser.add_argument('--model_dir', type=str, default='./models/')
parser.add_argument('--coeff', type=str, default=None) #I don't know what is this for

parser.add_argument('--shift', default=False, action="store_true", help='use shift for models')
parser.add_argument('--shift_div', default=8, type=int, help='number of div for shift (default: 8)')
parser.add_argument('--shift_place', default='blockres', type=str, help='place for shift (default: stageres)')
parser.add_argument('--temporal_pool', default=False, action="store_true", help='add temporal pooling')
parser.add_argument('--non_local', default=False, action="store_true", help='add non local block')
parser.add_argument('--dense_sample', default=False, action="store_true", help='use dense sample for video dataset')
parser.add_argument('--name', default='graph', type=str,
                    help='name of the model')

# # may contain splits
parser.add_argument('--weights', type=str, default=None)
parser.add_argument('--test_segments', type=str, default=25)

parser.add_argument('--twice_sample', default=False, action="store_true", help='use twice sample for ensemble')
parser.add_argument('--full_res', default=False, type=str2bool, help='Evaluate at full resolution')

parser.add_argument('--full_size_224', default=False, type=str2bool,
                    help='reschale to 224 crop the center 224x225')
parser.add_argument('--test_crops', type=int, default=1)
parser.add_argument('--test_batch_size', type=int, default=1)

# for true test
parser.add_argument('--test_list', type=str, default=None)
parser.add_argument('--csv_file', type=str, default=None)
parser.add_argument('--softmax', default=False, action="store_true", help='use softmax')
parser.add_argument('--max_num', type=int, default=-1)
parser.add_argument('--input_size', type=int, default=224)
parser.add_argument('--crop_fusion_type', type=str, default='avg')
parser.add_argument('--num_set_segments',type=int, default=1,help='TODO: select multiply set of n-frames from a video')

# ========================= Dynamic Regions Graph ==========================

parser.add_argument('--dynamic_regions', default='dyreg', type=str, choices=['none', 'pos_only', 'dyreg',
                    'semantic'],
                    help='type of regions used')              
parser.add_argument('--init_regions', default='grid', type=str, choices=['center', 'grid'], 
                    help='anchor position (default: center)')
parser.add_argument('--offset_lstm_dim', default=128, type=int, help='number channels for the offset lstm (default: 128)')
parser.add_argument('--use_rstg', default=False, type=str2bool,
                    help='divide regions size by a scaling factor')
parser.add_argument('--combine_by_sum', default=False, type=str2bool,
                    help='combine two vectors by concatenation or by summing')
parser.add_argument('--project_at_input', default=False, type=str2bool,
                    help='combine two vectors by concatenation or by summing')


parser.add_argument('--update_layers', type=int, default=2)
parser.add_argument('--send_layers', type=int, default=2)
parser.add_argument('--rnn_type', type=str, choices=['LSTM', 'GRU'], default='LSTM')
parser.add_argument('--aggregation_type', type=str, choices=['dot', 'sum'], default='dot')
parser.add_argument('--offset_generator', type=str, choices=['none', 'big', 'small'], default='big')

parser.add_argument('--place_graph', default='layer1.1_layer2.2_layer3.4_layer4.1', type=str,
                    help='where to place the graph: layeri.j_')
parser.add_argument('--rstg_combine', type=str, choices=['serial', 'plus'], default='plus')
parser.add_argument('--ch_div', type=int, default=2)
parser.add_argument('--graph_residual_type', default='norm', type=str,
                    help='norm / out_gate/ 1chan_out_gate/ gru_gate')
parser.add_argument('--remap_first', default=False, type=str2bool,
                    help='project and remap or remap and project')
parser.add_argument('--rstg_skip_connection', default=False, type=str2bool,
                    help='use skip connection from graphs')
parser.add_argument('--warmup_validate', default=False, type=str2bool,
                    help='warmup 100 steps before validate')                 
parser.add_argument('--tmp_norm_skip_conn', default=False, type=str2bool,
                    help='norm for skip connection')
parser.add_argument('--init_skip_zero', default=False, type=str2bool,
                    help='norm for skip connection')
parser.add_argument('--bottleneck_graph', default=False, type=str2bool,
                    help='smaller graph in bottleneck layer')


parser.add_argument('--eval_mode', default='test', type=str,
                    choices=['train', 'test'])
parser.add_argument('--visualisation', type=str, choices=['rgb', 'hsv'], default='hsv')
parser.add_argument('--save_kernels', default=False, type=str2bool,
                    help='save the kernels')   

#params for running distributed
parser.add_argument('--local_rank', type=int, default=0)
parser.add_argument('--ngpu', type=int, default=0)
parser.add_argument('--world_size', type=int, default=0)
parser.add_argument('--check_learned_params', default=False, type=str2bool)

def parse_args():
    from ops import models_config
    args = parser.parse_args()
    args.graph_params, args.out_pool_size, args.out_num_ch, args.distill_path = models_config.get_models_config()
    return args