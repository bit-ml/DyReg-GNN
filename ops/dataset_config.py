# Code adapted from "TSM: Temporal Shift Module for Efficient Video Understanding"
# Ji Lin*, Chuang Gan, Song Han


import os

global args, best_prec1
from opts import parse_args
args = parse_args()

if args.dataset == 'somethingv2':
    ROOT_DATASET = './data/smt-smt-V2/'  
elif args.dataset == 'something':
    ROOT_DATASET = './data/datasets/something-something/' 
elif args.dataset == 'others':
    ROOT_DATASET = './data/datasets/in_the_wild/'  
elif args.dataset == 'syncMNIST':
    ROOT_DATASET = './data/syncMNIST/'  
elif args.dataset == 'multiSyncMNIST':
    ROOT_DATASET = './data/multiSyncMNIST/' 


def return_something(modality):
    filename_categories = 'tsm_data/smtv1_category.txt'
    if modality == 'RGB':
        root_data = ROOT_DATASET + '/smt-smt-V2-frames/'
        filename_imglist_train = 'tsm_data/smtv1-train_videofolder.txt'
        filename_imglist_val = 'tsm_data/smtv1-val_videofolder.txt'
        prefix = '{:05d}.jpg'
    else:
        print('no such modality:'+modality)
        raise NotImplementedError
    return filename_categories, filename_imglist_train, filename_imglist_val, root_data, prefix


def return_others(modality):
    filename_categories = 'categories.txt' 

    root_data = ROOT_DATASET + '/gifs-frames-5s/'
    filename_imglist_train = 'gifs-frames-5s.txt'
    filename_imglist_val = 'gifs-frames-5s.txt'
    prefix = '{:06d}.jpg'

    return filename_categories, filename_imglist_train, filename_imglist_val, root_data, prefix



def return_somethingv2(modality):
    if modality == 'RGB':
        filename_categories = 'tsm_data/category.txt'
        root_data = ROOT_DATASET + '/smt-smt-V2-frames/'
        filename_imglist_train = 'tsm_data/train_videofolder.txt'
        filename_imglist_val = 'tsm_data/val_videofolder.txt'
        prefix = '{:06d}.jpg'
    else:
        print('no such modality:'+modality)
        raise NotImplementedError
    return filename_categories, filename_imglist_train, filename_imglist_val, root_data, prefix



def return_dataset(dataset, modality):
    if dataset == 'syncMNIST':
        n_class = 46
        test_dataset = '/data/datasets/video_mnist/sync_mnist_large_v2_split_test_max_sync_dist_160_num_classes_46_no_digits_5_no_noise_parts_0/'
        train_dataset = '/data/datasets/video_mnist/sync_mnist_large_v2_split_train_max_sync_dist_160_num_classes_46_no_digits_5_no_noise_parts_0/'
        return n_class, train_dataset, test_dataset, None, None
    elif dataset == 'multiSyncMNIST':
        n_class = 56
        test_dataset = args.test_dataset
        train_dataset = args.train_dataset
        return n_class, train_dataset, test_dataset, None, None 
    dict_single = {'something': return_something, 
                    'somethingv2': return_somethingv2,
                   'others' : return_others}
    if dataset in dict_single:
        file_categories, file_imglist_train, file_imglist_val, root_data, prefix = dict_single[dataset](modality)
    else:
        raise ValueError('Unknown dataset '+dataset)

    file_imglist_train = os.path.join(ROOT_DATASET, file_imglist_train)
    file_imglist_val = os.path.join(ROOT_DATASET, file_imglist_val)
   
    if isinstance(file_categories, str):
        file_categories = os.path.join(ROOT_DATASET, file_categories)
        with open(file_categories) as f:
            lines = f.readlines()
        categories = [item.rstrip() for item in lines]
    else:  # number of categories
        categories = [None] * file_categories
    n_class = len(categories)
    print('{}: {} classes'.format(dataset, n_class))
    return n_class, file_imglist_train, file_imglist_val, root_data, prefix
