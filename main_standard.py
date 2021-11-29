# Code adapted from "TSM: Temporal Shift Module for Efficient Video Understanding"
# Ji Lin*, Chuang Gan, Song Han

import os
import time
import shutil
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim


from ops.dataset import TSNDataSet
from ops.models import TSN
from ops.rstg import *
from ops.transforms import *
from ops import dataset_config
from ops.utils import LearnedParamChecker, AverageMeter, accuracy
from ops.utils import save_kernels,save_mean_kernels, count_params


from ops.temporal_shift import make_temporal_pool
import sys
from tensorboardX import SummaryWriter
import pdb
best_prec1 = 0
import pickle
import gc
from opts import parse_args

NGPU = torch.cuda.device_count() if torch.cuda.is_available() else 1
dev = torch.device('cuda', args.local_rank) if torch.cuda.is_available() else torch.device("cpu")

if args.dataset=='something':
    dataset = 'something-something'
elif args.dataset == 'somethingv2':
    dataset = 'smt-smt-V2'


def count_parameters(policies, model):
    learnable_params = []
    print_learnable_params = []
    for p in policies:
        list_name_param_tuple = p['params']
        list_params = []
        for name, param in list_name_param_tuple:
            learnable_params.append(name)
            print_learnable_params.append((name, param.shape))
            list_params.append(param)
        p['params'] = list_params
    print('_'*120)
    
    if args.local_rank == 0:
        print('Learnable parameters')
        print_learnable_params.sort()
        for name, shape in print_learnable_params:
            print(f'{name} shape: {shape}')

        print('NOT Learnable parameters')
        for name, param in model.named_parameters():
            name = name.replace('module.', '')
            if name not in learnable_params:
                print(f'{name} shape: {param.shape}')
        print('_'*120)

        print(f'Total number of params: {count_params(print_learnable_params)}')

        number_params = count_params(print_learnable_params, contains=['graph','norm_dict.residual_norm'])
        print(f'Graph number of params: {number_params}')

        number_params = count_params(print_learnable_params, contains=['rstg','norm_dict.residual_norm'])
        print(f'RSTG number of params: {number_params}')

        number_params = count_params(print_learnable_params, contains=['dynamic'], ignores=['rstg'])
        print(f'Dynamic number of params: {number_params}')
        print('_'*120)

    
def main():
    global args, best_prec1
    args = parse_args()

    print('Arguments:')
    for arg in vars(args):
        print(f'{arg}: {getattr(args, arg)}')

    num_class, args.train_list, args.val_list, args.root_path, prefix = dataset_config.return_dataset(args.dataset,
                                                                                                      args.modality)

    #prepare the args according to the current model
    full_arch_name = args.arch
    if args.shift:
        full_arch_name += '_shift{}_{}'.format(args.shift_div, args.shift_place)
    if args.temporal_pool:
        full_arch_name += '_tpool'
    args.store_name = '_'.join(
        ['TSM' + args.name, args.dataset, args.modality, full_arch_name, args.consensus_type, 'segment%d' % args.num_segments,
         'e{}'.format(args.epochs), f'lr_{args.lr}',f'batch_{args.batch_size}'])
    if args.pretrain != 'imagenet':
        args.store_name += '_{}'.format(args.pretrain)
    if args.lr_type != 'step':
        args.store_name += '_{}'.format(args.lr_type)
    if args.dense_sample:
        args.store_name += '_dense'
    if args.non_local > 0:
        args.store_name += '_nl'
    if args.suffix is not None:
        args.store_name += '_{}'.format(args.suffix)
    with open(os.path.join(args.model_dir, args.store_name + '_args.txt'), 'w') as f:
        f.write(str(args))
    args.store_name = ''
    print('storing name: ' + args.store_name)

    if args.local_rank == 0:
        check_rootfolders()

    if torch.cuda.is_available():
        torch.cuda.set_device(args.local_rank)
        world_size = NGPU
        torch.distributed.init_process_group(
                                    backend='nccl',
                                    init_method='env://',
                                    world_size=world_size, 
                                    rank=args.local_rank)

    model = TSN(num_class, args.num_segments, args.modality,
                base_model=args.arch,
                consensus_type=args.consensus_type,
                dropout=args.dropout,
                img_feature_dim=args.img_feature_dim,
                partial_bn=not args.no_partialbn,
                pretrain=args.pretrain,
                is_shift=args.shift, shift_div=args.shift_div, shift_place=args.shift_place,
                fc_lr5=not (args.tune_from and args.dataset in args.tune_from),
                temporal_pool=args.temporal_pool,
                non_local=args.non_local)

    if torch.cuda.is_available():
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    crop_size = model.crop_size
    scale_size = model.scale_size
    input_mean = model.input_mean
    input_std = model.input_std
   
    policies = model.get_optim_policies()
    train_augmentation = model.get_augmentation(flip=False if 'something' in args.dataset else True)

    if torch.cuda.is_available():
        model = model.to(dev)
        model = torch.nn.parallel.DistributedDataParallel(
                                            model,
                                            device_ids=[args.local_rank],
                                            output_device=args.local_rank,
                                            find_unused_parameters=True,
                                        )
    #count and print learnable parameters 
    count_parameters(policies, model)

    optimizer = torch.optim.SGD(policies,
                            args.lr,
                            momentum=args.momentum,
                            weight_decay=args.weight_decay)

    #restore parameters from a checkpoint
    if args.resume:
        if args.temporal_pool:  
            make_temporal_pool(model.module.base_model, args.num_segments)
        if os.path.isfile(args.resume):
            print(("=> loading checkpoint '{}'".format(args.resume)))
            checkpoint =torch.load(args.resume, map_location=lambda storage, loc: storage) 


            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            print(args.start_epoch)
            ckpt_dict = checkpoint['state_dict']
            model_dict = model.state_dict()
            print('checkpoint parameters')
            for k, v in ckpt_dict.items():
                print(f'{k} shape: {v.shape}')

            restore_dict = {}
            for k, v in ckpt_dict.items():
                k = k.replace('.block','.0.block')
                k = k.replace('.dynamic_graph.','.1.dynamic_graph.')
                k = k.replace('.norm_dict.residual_norm','.1.norm_dict.residual_norm')
                if not torch.cuda.is_available():
                    k = k.replace('module.', '')
        
                # don't load constants
                if ('dynamic_graph.ph' not in k 
                        and 'dynamic_graph.pw' not in k 
                        and 'dynamic_graph.arange_h' not in k 
                        and 'dynamic_graph.arange_w' not in k
                ):
                    restore_dict[k] = v
                if 'base_model.map_final_project' in k:
                    new_name = k.replace('base_model.', '')
                    restore_dict[new_name] = v
                    del restore_dict[k]
                
    
            if args.replace_ignore:
                for k, v in model_dict.items():
                    if 'ignore' in k:
                        old_name = k.replace('ignore.', '')
                        ckpt_val = ckpt_dict[old_name]
                        del restore_dict[old_name]
                        restore_dict[k] = ckpt_val     
            model_dict.update(restore_dict)
            model.load_state_dict(model_dict)
            optimizer.load_state_dict(checkpoint['optimizer'])
            print(("=> loaded checkpoint '{}' (epoch {})"
                   .format(args.evaluate, checkpoint['epoch'])))
        else:
            print(("=> no checkpoint found at '{}'".format(args.resume)))
    
    #restore subset of parameters from an existing checkpoint
    elif args.tune_from:
        if args.temporal_pool:  
            make_temporal_pool(model.module.base_model, args.num_segments)
        if os.path.isfile(args.tune_from):
            print(("=> loading checkpoint '{}'".format(args.tune_from)))
            checkpoint =torch.load(args.tune_from, map_location=lambda storage, loc: storage) 
    
            
            ckpt_dict = checkpoint['state_dict']
            model_dict = model.state_dict()

            restore_dict = {}
            for k, v in ckpt_dict.items():
                if ('dynamic_graph.ph' not in k 
                        and 'dynamic_graph.pw' not in k 
                        and 'dynamic_graph.arange_h' not in k 
                        and 'dynamic_graph.arange_w' not in k
                        and 'new_fc' not in k # do not restore last fc
                ):
                    restore_dict[k] = v
                if 'base_model.map_final_project' in k:
                    new_name = k.replace('base_model.', '')
                    restore_dict[new_name] = v
                    del restore_dict[k]

            for k, v in model_dict.items():
                if 'ignore' in k:
                    old_name = k.replace('ignore.', '')
                    ckpt_val = ckpt_dict[old_name]
                    del restore_dict[old_name]
                    restore_dict[k] = ckpt_val      
            model_dict.update(restore_dict)
            model.load_state_dict(model_dict)
           
            print(("=> loaded checkpoint '{}' (epoch {})"
                   .format(args.evaluate, checkpoint['epoch'])))
        else:
            print(("=> no checkpoint found at '{}'".format(args.resume)))

    else:
        if args.temporal_pool:
            make_temporal_pool(model.module.base_model, args.num_segments)

    cudnn.benchmark = True
    
    #Load data
    normalize = GroupNormalize(input_mean, input_std)
    data_length = 1

    train_dataset = TSNDataSet(args.root_path, args.train_list, num_segments=args.num_segments,
                            new_length=data_length,
                            modality=args.modality,
                            image_tmpl=prefix,
                            transform=torchvision.transforms.Compose([
                                train_augmentation,
                                Stack(roll=(False)),
                                ToTorchFormatTensor(div=(True)),
                                normalize,
                            ]), dense_sample=args.dense_sample, split='train')

    train_sampler = torch.utils.data.distributed.DistributedSampler(
                                                    train_dataset,
                                                    num_replicas=NGPU,
                                                    rank=args.local_rank,
                                                    shuffle=True,
                                                    ) if torch.cuda.is_available() else None
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size, 
        num_workers=(args.workers if torch.cuda.is_available() else 0), pin_memory=True,
        sampler=train_sampler,
        drop_last=True)  # prevent something not % n_GPU
    if not torch.cuda.is_available():
        train_loader.shuffle=True

    val_dataset = TSNDataSet(args.root_path, args.val_list, num_segments=args.num_segments,
                                    new_length=data_length,
                                    modality=args.modality,
                                    image_tmpl=prefix,
                                    random_shift=False,
                                    transform=torchvision.transforms.Compose([
                                        GroupScale(int(scale_size)),
                                        GroupCenterCrop(crop_size),
                                        Stack(roll=(False)),
                                        ToTorchFormatTensor(div=(True)),
                                        normalize,
                                    ]), dense_sample=args.dense_sample, split='val')
    val_sampler = torch.utils.data.distributed.DistributedSampler(
                                                        val_dataset,
                                                        num_replicas=NGPU,
                                                        rank=args.local_rank,
                                                        shuffle=False
                                                        )if torch.cuda.is_available() else None
    val_loader = torch.utils.data.DataLoader(
                        val_dataset,
                        sampler=val_sampler,
                        batch_size=args.batch_size,
                        num_workers=(args.workers if torch.cuda.is_available() else 0), pin_memory=True, drop_last=True)
    if not torch.cuda.is_available():
        val_loader.shuffle=False

    criterion = torch.nn.CrossEntropyLoss().to(dev)

    for group in policies:
        print(('group: {} has {} params, lr_mult: {}, decay_mult: {}'.format(
            group['name'], len(group['params']), group['lr_mult'], group['decay_mult'])))

    # only master thread
    log_training = None
    tf_writer = None
    if args.local_rank==0:
        log_training = open(os.path.join(args.model_dir, args.store_name, args.root_log, 'log.csv'), 'w')
        with open(os.path.join(args.model_dir, args.store_name, args.root_log, 'args.txt'), 'w') as f:
            f.write(str(args))
        tf_writer = SummaryWriter(log_dir=os.path.join(args.model_dir, args.store_name, args.root_log))

    if args.evaluate:
        prec1 = validate(val_loader, model, criterion, args.start_epoch-1, log_training, tf_writer, max_iters=None)
        tf_writer.close()
        return

    prec1 = validate(val_loader, model, criterion, args.start_epoch-1, log_training, tf_writer, max_iters=None)
    save_checkpoint({
            'epoch': args.start_epoch,
            'arch': args.arch,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'best_prec1': best_prec1,
        }, False, aux_name='initial_model')

    # save initial parameters:
    param_checker = None
    if args.check_learned_params:
        param_checker = LearnedParamChecker(model)
    
            
    for epoch in range(args.start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch, args.lr_type, args.lr_steps)
        
        if torch.cuda.is_available():
            train_sampler.set_epoch(epoch)
        # train for one epoch
        train(train_loader, model, criterion,  optimizer, epoch, log_training, tf_writer,param_checker, max_iters=None)

        if args.check_learned_params:
            param_checker.compare_current_initial_params()
     

        # evaluate on validation set
        if (epoch + 1) % args.eval_freq == 0 or epoch == args.epochs - 1:
            prec1 = validate(val_loader, model, criterion, epoch, log_training, tf_writer, max_iters=None)

            # remember best prec@1 and save checkpoint
            is_best = prec1 > best_prec1
            best_prec1 = max(prec1, best_prec1)

            if args.local_rank == 0:
                tf_writer.add_scalar('acc/test_top1_best', best_prec1, epoch)

                output_best = 'Best Prec@1: %.3f\n' % (best_prec1)
                print(output_best)
                log_training.write(output_best + '\n')
                log_training.flush()

                # Overwrite last checkpoint at each epoch
                save_checkpoint({
                    'epoch': epoch + 1,
                    'arch': args.arch,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'best_prec1': best_prec1,
                }, is_best)

                if (epoch + 1) % (5 * args.eval_freq) == 0 or epoch == args.epochs - 1:
                    save_checkpoint({
                        'epoch': epoch + 1,
                        'arch': args.arch,
                        'state_dict': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'best_prec1': best_prec1,
                    }, is_best, aux_name=f'epoch_{epoch}')


# from crop dimension to image dimension
def transform(point, dim, crop_trans):
    # crop_trans: (resize_ratio_h, resize_ratio_w, offset_h, offset_w)
    abs_dh = point[:,:,0] / crop_trans[:,:,0] + crop_trans[:,:,2]
    abs_dw = point[:,:,1] / crop_trans[:,:,1] + crop_trans[:,:,3]
    abs_h = dim[:,:,0] / crop_trans[:,:,0]
    abs_w = dim[:,:,1] / crop_trans[:,:,1]
    # return [float(abs_dh.cpu().numpy()), float(abs_dw.cpu().numpy()), float(abs_h.cpu().numpy()), float(abs_w.cpu().numpy())]
    return np.stack([abs_h, abs_w, abs_dh, abs_dw], axis=-1)

def train(train_loader, model, criterion,optimizer, epoch, log, tf_writer, param_checker, max_iters=None):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    top1 = AverageMeter()
    top5 = AverageMeter()

    if torch.cuda.is_available():
        if args.no_partialbn:
            model.module.partialBN(False)
        else:
            model.module.partialBN(True)
    else:
        if args.no_partialbn:
            model.partialBN(False)
        else:
            model.partialBN(True)

    list_save_iter = list(np.linspace(0, len(train_loader) // 1, 5 ).astype(int))[:4]

    # switch to train mode
    model.train()
    end = time.time()
    
   
    print(f'Epoch: {epoch}')
    for i, (_, input, target, _) in enumerate(train_loader):
        if args.check_learned_params:
            if i % 10 == 0:
                param_checker.compare_current_initial_params()
          
        if max_iters is not None:
            if i > max_iters:
                break
        # measure data loading time
        data_time.update(time.time() - end)

        target = target.to(dev)
        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target)
        
        
        output, model_aux_feats = model(input_var)
        offsets = model_aux_feats['offsets']
        aux = model_aux_feats['interm_feats']

        loss = criterion(output, target_var)
        
        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))
        top5.update(prec5.item(), input.size(0))

        # compute gradient and do SGD step
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if args.use_rstg and i in list_save_iter and args.local_rank==0:
            folder = os.path.join(args.model_dir, args.store_name, args.root_log,'kernels')
            save_kernels(input, aux, folder=folder, name=f'train_epoch{epoch}_iter_{i}_{args.local_rank}', predicted_offsets=offsets)

        if i % args.print_freq == 0:
            if torch.cuda.is_available():
                # args.distributed:
                effective_batch_size = torch.cuda.device_count() * args.batch_size
            else:
                effective_batch_size = args.batch_size
            
            output = ('[rank {rank}] Epoch: [{0}][{1}/{2}], lr: {lr:.5f}\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                epoch, i, len(train_loader) // (1 if args.dataset != 'syncMNIST' else effective_batch_size),
                batch_time=batch_time,
                data_time=data_time, loss=losses, top1=top1, top5=top5, rank=args.local_rank,lr=optimizer.param_groups[-3]['lr']))  # TODO
            
            print(output)
            sys.stdout.flush()
            
            if args.local_rank==0:
                log.write(output + '\n')
                log.flush()

        if i % 100 == 0:
            gc.collect()

    if args.local_rank==0:
        tf_writer.add_scalar('loss/train', losses.avg, epoch)
        tf_writer.add_scalar('acc/train_top1', top1.avg, epoch)
        tf_writer.add_scalar('acc/train_top5', top5.avg, epoch)
        tf_writer.add_scalar('lr', optimizer.param_groups[-1]['lr'], epoch)
 
def save_data(data, filename):
    with open(filename + '.pickle', 'wb') as f:
        pickle.dump(data,f)
    
    with open(filename + '.txt', 'w') as f:
        if isinstance(data, dict):
            for k,v in data.items():
                f.write(f'{k}: {v}\n')
        else:
            f.write(str(data))
    
def validate(val_loader, model, criterion, epoch, log=None, tf_writer=None, max_iters=None):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # warmup BN statistics
    if args.warmup_validate == True:
        warm_max_iters = 500
        with torch.no_grad():
            for i, (input, target) in enumerate(val_loader):
                if i % 20 == 0:
                    print(f'Warmup iter: {i}')

                if i > warm_max_iters:
                    break
                output, _ = model(input)

    model.eval()
    # TODO: de un-hardcodat
    E_kernels = {'layer2_block2' : 0, 'layer3_block4' : 0, 'layer4_block1' : 0}
    if args.arch == 'resnet18':
        E_kernels = {'layer2_block1' : 0, 'layer3_block1' : 0, 'layer4_block1' : 0}
    elif args.arch == 'resnet34':
        E_kernels = {'layer2_block2' : 0, 'layer3_block4' : 0, 'layer4_block1' : 0}
    elif args.arch == 'wide_resnet50_2':
        E_kernels = {'layer2_block2' : 0, 'layer3_block1' : 0, 'layer4_block1' : 0}
    elif args.arch == 'resnet101':
        E_kernels = {'layer2_block2' : 0, 'layer3_block1' : 0, 'layer3_block4' : 0, 'layer4_block1' : 0}


    # E_kernels = {'layer3_block4' : 0}

    nr_batches = 0
    end = time.time()
    
    with torch.no_grad():
        for i, (gt_boxes, input, target, detected_boxes) in enumerate(val_loader):
            time1 = time.time()
            if max_iters is not None:
                if i > max_iters:
                    break
            target = target.to(dev)
            output, model_aux_feats = model(input)

            offsets = model_aux_feats['offsets']
            aux = model_aux_feats['interm_feats']

            # add all kernels
            nr_batches += 1
            if args.use_rstg:
                for key, val in model_aux_feats['kernel'].items():
                    c_kernel = model_aux_feats['kernel'][key]
                    c_kernel = c_kernel.view(args.batch_size, args.num_segments,9,c_kernel.shape[-2], c_kernel.shape[-1])
                    c_kernel = c_kernel.detach().cpu().numpy().mean(0)
                    E_kernels[key] = E_kernels[key] + c_kernel

            if args.dataset == 'syncMNIST':
                loss = criterion(output, target.view(target.shape[0]))
            else:
                loss = criterion(output, target)
            
 
            # measure accuracy and record loss
            prec1, prec5 = accuracy(output.data, target, topk=(1, 5))

            losses.update(loss.item(), input.size(0))
            top1.update(prec1.item(), input.size(0))
            top5.update(prec5.item(), input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            
            folder = os.path.join(args.model_dir, args.store_name, args.root_log,'kernels')
            #if i % args.print_freq == 0:
            if args.dataset == 'others' or (args.use_rstg and i in [10,100,1000,2000,3000,4000,5000]):# and not args.evaluate:
                save_kernels(input, aux, folder=folder, name=f'valid_epoch_{epoch}_iter_{i}_{args.local_rank}', target_boxes_val=gt_boxes, predicted_offsets=offsets)

            if i % args.print_freq == 0:
                output = ('[rank {rank}] Test: [{0}/{1}]\t'
                          'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                          'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                          'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                          'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                    i, len(val_loader)// (1 if args.dataset != 'syncMNIST' else args.batch_size),
                    batch_time=batch_time, loss=losses, rank=args.local_rank,
                    top1=top1, top5=top5))
                print(output)

                if args.local_rank == 0 and log is not None:
                    log.write(output + '\n')
                    log.flush()

    if args.use_rstg:
        for key, _ in model_aux_feats['kernel'].items():
            E_kernels[key] /= nr_batches

        save_mean_kernels(E_kernels, epoch=epoch, folder=folder)

    output = ('Process {local_rank} Individual Testing Results: Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f} Loss {loss.avg:.5f}'
              .format(top1=top1, top5=top5, loss=losses, local_rank=args.local_rank))
    print(output)

    top1_avg = torch.tensor(top1.avg).to(dev)
    top5_avg = torch.tensor(top5.avg).to(dev)
    losses_avg = torch.tensor(losses.avg).to(dev)

    if torch.cuda.is_available():
        torch.distributed.reduce(top1_avg, dst=0) 
        torch.distributed.reduce(top5_avg, dst=0) 
        torch.distributed.reduce(losses_avg, dst=0) 
    
    top1_avg = top1_avg / NGPU
    top5_avg = top5_avg / NGPU
    losses_avg = losses_avg / NGPU

    
    if args.local_rank == 0:
        all_output = ('[Reduced] Testing Results: Prec@1 {top1:.3f} Prec@5 {top5:.3f} Loss {loss:.5f}'
              .format(top1=top1_avg, top5=top5_avg, loss=losses_avg))
        print(all_output)
        if log is not None:
            log.write(output + '\n')
            log.flush()
            tf_writer.add_scalar('loss/test', losses_avg, epoch)
            tf_writer.add_scalar('acc/test_top1', top1_avg, epoch)
            tf_writer.add_scalar('acc/test_top5', top5_avg, epoch)
    
    return top1_avg


def save_checkpoint(state, is_best, aux_name=''):
    # filename = '%s/%s/ckpt.pth.tar' % (args.model_dir, args.store_name, args.root_model)
    filename = f'{args.model_dir}/{args.store_name}/{ args.root_model}/{aux_name}ckpt.pth.tar'
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, filename.replace(f'{aux_name}ckpt.pth.tar', 'best.pth.tar'))


def adjust_learning_rate(optimizer, epoch, lr_type, lr_steps):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    if lr_type == 'step':
        decay = 0.1 ** (sum(epoch >= np.array(lr_steps)))
        lr = args.lr * decay
        decay = args.weight_decay
    elif lr_type == 'small_init_step':
        if epoch < 10:
            lr = args.lr * 0.2
            decay = args.weight_decay
        else:
            decay = 0.1 ** (sum(epoch >= np.array(lr_steps)))
            lr = args.lr * decay
            decay = args.weight_decay
    elif lr_type == 'cos':
        import math
        lr = 0.5 * args.lr * (1 + math.cos(math.pi * epoch / args.epochs))
        decay = args.weight_decay
    else:
        raise NotImplementedError
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr * param_group['lr_mult']
        param_group['weight_decay'] = decay * param_group['decay_mult']


def check_rootfolders():
    """Create log and model folder"""
    folders_util = [args.model_dir, args.root_log, args.root_model,
                    os.path.join(args.model_dir, args.store_name, args.root_log),
                    os.path.join(args.model_dir, args.store_name, args.root_model)]
    for folder in folders_util:
        if not os.path.exists(folder):
            print('creating folder ' + folder)
            os.makedirs(folder)



if __name__ == '__main__':
    main()
