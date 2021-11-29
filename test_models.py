# Code for "TSM: Temporal Shift Module for Efficient Video Understanding"
# arXiv:1811.08383
# Ji Lin*, Chuang Gan, Song Han
# {jilin, songhan}@mit.edu, ganchuang@csail.mit.edu

# Notice that this file has been modified to support ensemble testing

import argparse
import time

import torch.nn.parallel
import torch.optim
from sklearn.metrics import confusion_matrix
from ops.dataset import TSNDataSet
from ops.models import TSN
from ops.transforms import *
from ops import dataset_config
from torch.nn import functional as F
import pdb
# options
import os
from opts import parser
from ops.utils import save_kernels

global args, best_prec1
args = parser.parse_args()
print(args)

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    res = []
    for k in topk:
        #  correct_k = correct[:k].view(-1).float().sum(0)
        correct_k = correct[:k].float().sum()
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def parse_shift_option_from_log_name(log_name):
    if 'shift' in log_name:
        strings = log_name.split('_')
        for i, s in enumerate(strings):
            if 'shift' in s:
                break
        return True, int(strings[i].replace('shift', '')), strings[i + 1]
    else:
        return False, None, None


weights_list = args.weights.split(',')
test_segments_list = [int(s) for s in args.test_segments.split(',')]
assert len(weights_list) == len(test_segments_list)
if args.coeff is None:
    coeff_list = [1] * len(weights_list)
else:
    coeff_list = [float(c) for c in args.coeff.split(',')]

if args.test_list is not None:
    test_file_list = args.test_list.split(',')
else:
    test_file_list = [None] * len(weights_list)


data_iter_list = []
net_list = []
modality_list = []

total_num = None
for this_weights, this_test_segments, test_file in zip(weights_list, test_segments_list, test_file_list):
    is_shift, shift_div, shift_place = args.shift, args.shift_div, args.shift_place
    modality = args.modality
    #this_arch = this_weights.split('TSM_')[1].split('_')[2]
    this_arch = args.arch
    modality_list.append(modality)
    num_class, args.train_list, val_list, root_path, prefix = dataset_config.return_dataset(args.dataset,
                                                                                            modality)
    print('=> shift: {}, shift_div: {}, shift_place: {}'.format(is_shift, shift_div, shift_place))
    net = TSN(num_class, this_test_segments if is_shift else 1, modality,
              base_model=this_arch,
              consensus_type=args.crop_fusion_type,
              dropout=args.dropout,
              img_feature_dim=args.img_feature_dim,
              partial_bn=not args.no_partialbn,
              fc_lr5=not (args.tune_from and args.dataset in args.tune_from),
              pretrain=args.pretrain,
              is_shift=is_shift, shift_div=shift_div, shift_place=shift_place,
              temporal_pool=args.temporal_pool,
              non_local='_nl' in this_weights,
              )

    if 'tpool' in this_weights:
        from ops.temporal_shift import make_temporal_pool
        make_temporal_pool(net.base_model, this_test_segments)  # since DataParallel

    print(f'Loading weights from: {this_weights}')
    checkpoint = torch.load(this_weights, map_location=torch.device('cpu'))
    epoch = checkpoint['epoch']
    checkpoint = checkpoint['state_dict']    
    print(f'Evaluating epoch: {epoch}')
    ckpt_dict = {}
    for k, v in list(checkpoint.items()):
        if ('dynamic_graph.ph' not in k and 'dynamic_graph.pw' not in k and 'dynamic_graph.arange_h' not in k and 'dynamic_graph.arange_w' not in k
                     and 'const_dh_ones' not in k and 'const_dw_ones' not in k and 'fix_offsets' not in k):
            # remove first tag ('module')
            key_name = '.'.join(k.split('.')[1:])
            # remove tags from checkpoints vars
            key_name = key_name.replace('base_model.map_final_project','map_final_project')
            key_name = key_name.replace('.block','.0.block')
            key_name = key_name.replace('.dynamic_graph.','.1.dynamic_graph.')
            key_name = key_name.replace('.norm_dict.residual_norm','.1.norm_dict.residual_norm')
            ckpt_dict[key_name] = v

    model_dict = net.state_dict()

    print('Model parameters')
    [print(k) for k in model_dict.keys()]
    
    print('checkpoint parameters')
    [print(k) for k in checkpoint.keys()]
    
    # 
    for k, v in model_dict.items():
        if 'ignore' in k:
            old_name = k.replace('ignore.', '')
            ckpt_val = checkpoint['module.' +old_name]
            del ckpt_dict[old_name]
            ckpt_dict[k] = ckpt_val      

    model_dict.update(ckpt_dict)
    net.load_state_dict(model_dict)

    input_size = net.scale_size if args.full_res else net.input_size


    scale1 = net.scale_size
    crop1 = input_size
    if args.test_crops == 1:
        cropping = torchvision.transforms.Compose([
            GroupScale(scale1),
            GroupCenterCrop(crop1),
        ])
    elif args.test_crops == 3:  # do not flip, so only 5 crops
        cropping = torchvision.transforms.Compose([
            GroupFullResSample(input_size, net.scale_size, flip=False)
        ])
    elif args.test_crops == 5:  # do not flip, so only 5 crops
        cropping = torchvision.transforms.Compose([
            GroupOverSample(input_size, net.scale_size, flip=False)
        ])
    elif args.test_crops == 10:
        cropping = torchvision.transforms.Compose([
            GroupOverSample(input_size, net.scale_size)
        ])
    else:
        raise ValueError("Only 1, 5, 10 crops are supported while we got {}".format(args.test_crops))
    data_loader = torch.utils.data.DataLoader(
            TSNDataSet(root_path, test_file if test_file is not None else val_list, num_segments=this_test_segments,
                       new_length=1,
                       modality=modality,
                       image_tmpl=prefix,
                       test_mode=True,
                       remove_missing=len(weights_list) == 1,
                       transform=torchvision.transforms.Compose([
                           cropping,
                           Stack(roll=(False)),
                           ToTorchFormatTensor(div=(True)),
                           GroupNormalize(net.input_mean, net.input_std),
                       ]), dense_sample=args.dense_sample, twice_sample=args.twice_sample),
            batch_size=args.batch_size, shuffle=False,
            num_workers=args.workers if torch.cuda.is_available() else 0, pin_memory=True
    )

    if args.gpus is not None:
        devices = [args.gpus[i] for i in range(args.workers)]
    else:
        devices = list(range(args.workers))

    if torch.cuda.is_available():
        net = torch.nn.DataParallel(net.cuda())
    net.eval()
    data_gen = enumerate(data_loader)
    if total_num is None:
        total_num = len(data_loader.dataset)
    else:
        assert total_num == len(data_loader.dataset)
    data_iter_list.append(data_gen)
    net_list.append(net)
output = []

def eval_video(video_data, net, this_test_segments, modality, mode='eval'):
    if mode == 'eval':
        net.eval()
    else:
        print("something wrong. double check the flags")
        net.train()
    with torch.no_grad():
        i, data, label, _ = video_data
        batch_size = label.numel()
        num_crop = args.test_crops
        if args.dense_sample:
            num_crop *= 10  # 10 clips for testing when using dense sample

        if args.twice_sample:
            num_crop *= 2

        length = 3
        data_in = data

        if is_shift:
            data_in = data_in.view(batch_size * num_crop, this_test_segments, length, data_in.size(2), data_in.size(3))

        
        rst, model_aux_feats = net(data_in)
        aux = model_aux_feats['interm_feats']
        offsets = model_aux_feats['offsets']
        
        list_save_iter = [0,100,1000]
        if args.save_kernels and i in list_save_iter: 
            folder = os.path.join(args.model_dir, args.store_name, args.root_log,'kernels')
            save_kernels(data_in, aux, folder=folder, name=f'validation_iter_{i}', predicted_offsets=offsets)
        rst = rst.reshape(batch_size, num_crop, -1).mean(1)

        if args.softmax:
            # take the softmax to normalize the output to probability
            rst = F.softmax(rst, dim=1)
        rst = rst.data.cpu().numpy().copy()

        if torch.cuda.is_available():
            if net.module.is_shift:
                rst = rst.reshape(batch_size, num_class)
            else:
                rst = rst.reshape((batch_size, -1, num_class)).mean(axis=1).reshape((batch_size, num_class))
        else:
            if net.is_shift:
                rst = rst.reshape(batch_size, num_class)
            else:
                rst = rst.reshape((batch_size, -1, num_class)).mean(axis=1).reshape((batch_size, num_class))
        return i, rst, label


proc_start_time = time.time()
max_num = args.max_num if args.max_num > 0 else total_num

top1 = AverageMeter()
top5 = AverageMeter()

all_batch_time = [] 
for i, data_label_pairs in enumerate(zip(*data_iter_list)):
    with torch.no_grad():
        if i >= max_num:
            break
        this_rst_list = []
        this_label = None
        begin_proc = time.time()
        for n_seg, (_, (_, data, label,detector_out)), net, modality in zip(test_segments_list, data_label_pairs, net_list, modality_list):
            rst = eval_video((i, data, label,detector_out), net, n_seg, modality, mode='eval')
            this_rst_list.append(rst[1])
            this_label = label
        assert len(this_rst_list) == len(coeff_list)
        for i_coeff in range(len(this_rst_list)):
            this_rst_list[i_coeff] *= coeff_list[i_coeff]
        ensembled_predict = sum(this_rst_list) / len(this_rst_list)

        for p, g in zip(ensembled_predict, this_label.cpu().numpy()):
            output.append([p[None, ...], g])
        cnt_time = time.time() - proc_start_time
        batch_time = time.time() - begin_proc
        if i > 0:
            all_batch_time.append(batch_time)
        prec1, prec5 = accuracy(torch.from_numpy(ensembled_predict), this_label, topk=(1, 5))
        top1.update(prec1.item(), this_label.numel())
        top5.update(prec5.item(), this_label.numel())
        if i % 5 == 0:
            print('video {} done, total {}/{}, average {:.3f} sec/video, '
                  'moving Prec@1 {:.3f} Prec@5 {:.3f}'.format(i * args.batch_size, i * args.batch_size, total_num,
                                                              float(cnt_time) / (i+1) / args.batch_size, top1.avg, top5.avg))
            print(f"average sec/video: {np.array(all_batch_time).mean() / args.batch_size} ")
video_pred = [np.argmax(x[0]) for x in output]
video_pred_top5 = [np.argsort(np.mean(x[0], axis=0).reshape(-1))[::-1][:5] for x in output]

video_labels = [x[1] for x in output]


if args.csv_file is not None:
    print('=> Writing result to csv file: {}'.format(args.csv_file))
    with open(test_file_list[0].replace('test_videofolder.txt', 'category.txt')) as f:
        categories = f.readlines()
    categories = [f.strip() for f in categories]
    with open(test_file_list[0]) as f:
        vid_names = f.readlines()
    vid_names = [n.split(' ')[0] for n in vid_names]
    assert len(vid_names) == len(video_pred)
    if args.dataset != 'somethingv2':  # only output top1
        with open(args.csv_file, 'w') as f:
            for n, pred in zip(vid_names, video_pred):
                f.write('{};{}\n'.format(n, categories[pred]))
    else:
        with open(args.csv_file, 'w') as f:
            for n, pred5 in zip(vid_names, video_pred_top5):
                fill = [n]
                for p in list(pred5):
                    fill.append(p)
                f.write('{};{};{};{};{};{}\n'.format(*fill))


cf = confusion_matrix(video_labels, video_pred).astype(float)

np.save('cm.npy', cf)
cls_cnt = cf.sum(axis=1)
cls_hit = np.diag(cf)
eps = 0.0000001
cls_acc = cls_hit / (cls_cnt + eps)
print(cls_acc)
upper = np.mean(np.max(cf, axis=1) / (cls_cnt + eps))
print('upper bound: {}'.format(upper))

print('-----Evaluation is finished------')
print('Class Accuracy {:.02f}%'.format(np.mean(cls_acc) * 100))
print('Overall Prec@1 {:.02f}% Prec@5 {:.02f}%'.format(top1.avg, top5.avg))


