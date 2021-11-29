import numpy as np
from glob import glob
import os
import pickle
import torch
import time
import pdb
import math


def get_labels_dict_multi_sincron():
    # create labels
    label = 0
    labels_dict = {}
    for i in range(10):
        for j in range(i):
            labels_dict[(i, j)] = label
            labels_dict[(j, i)] = label
            label += 1
    # no sync digits
    labels_dict[(-1, -1)] = label
    label += 1
    # 
    for i in range(10):
        labels_dict[(i, i)] = label
        label += 1

    return labels_dict, label

class SyncedMNISTDataSet(torch.utils.data.IterableDataset):
    def __init__(self, split='train', dataset_path=None, dataset_fraction_used=1.0):
        super(SyncedMNISTDataSet).__init__()
        #assert end > start, "this example code only works with end >= start"
        self.split = split
        #self.dataset_path = f'/data/datasets/video_mnist/sync_mnist_large_v2_split_{split}_max_sync_dist_160_num_classes_46_no_digits_5_no_noise_parts_0_uint8/'
        #self.dataset_path = f'/data/datasets/video_mnist/sync_mnist_large_v2_split_{split}_max_sync_dist_160_num_classes_46_no_digits_5_no_noise_parts_0_uint8/'
        #self.dataset_path = f'/data/datasets/video_mnist/sync_mnist_large_v2_split_{split}_max_sync_dist_160_num_classes_46_no_digits_5_no_noise_parts_0_uint8/'
        self.dataset_path = dataset_path
        self.worker_id = 0
        self.max_workers = 0
        self.dataset_len = len(glob(self.dataset_path + '/data*pickle')) * 1000
        self.dataset_fraction_used = dataset_fraction_used
        # print(f'in iterable dataset: {self.worker_id } / {self.max_workers}')
        self.gen = None

    def __iter__(self):
        # print(f'worker: [{self.worker_id } / {self.max_workers}]')
        return self.gen.next_item(self.worker_id)
    def __len__(self):
        return self.dataset_len
def syncMNIST_worker_init_fn(worker_id):
    worker_info = torch.utils.data.get_worker_info()
    dataset = worker_info.dataset  # the dataset copy in this worker process
    dataset.worker_id = worker_info.id
    dataset.max_workers = worker_info.num_workers
    # print(f'in woker init fct: {dataset.worker_id}/ {dataset.max_workers}')
    dataset.gen = SyncMNISTGenerator(dataset_path=dataset.dataset_path,
            worker_id=dataset.worker_id, max_workers=dataset.max_workers,
            split=dataset.split, dataset_fraction_used=dataset.dataset_fraction_used)
    # print(f'in woker init fct: len gen {len(dataset.gen)}')

def read_data_mnist(file, num_classes=65):
    with open(file, 'rb') as fo:
        # print(f'loading: {file}')
        videos_dict = pickle.load(fo)
        # x = np.expand_dims(videos_dict['videos'], 4)
        # x = videos_dict['videos']#.astype(np.float32) / 255.0 * 2 - 1.0
        x = videos_dict['videos']#.astype(np.uint8)

        y = videos_dict['labels'].astype(int).squeeze()
        y = np.clip(y, 0, num_classes-1)
        #y = np.expand_dims(np.eye(num_classes)[y], axis=1)
        # y = y.astype(np.float32)

        coords = videos_dict['videos_digits_coords']
        top_left    = coords
        bot_right   = coords + 28
        digits_boxes = np.concatenate([top_left,bot_right], axis=-1) // 2
        is_ann_boxes = np.ones((digits_boxes.shape[0], 1))
        
        video = x
        label = y.astype(np.int64)

        if False:
            digits = videos_dict['videos_digits']
            min_digits = digits.min(1)
            max_digits = digits.max(1)
            labels_dict, max_labels = get_labels_dict_multi_sincron()

            min_max = np.zeros_like(label)
            for i in range(min_max.shape[0]):
                min_max[i] = labels_dict[(min_digits[i], max_digits[i])]
            

        # print(f'video.dtype: {video.dtype}')
        # video = torch.from_numpy(x).float()
        # label = torch.from_numpy(y).long()
        # print(f'video: {video.shape}')
    return video, label, digits_boxes, is_ann_boxes
    # return video, label, digits_boxes, is_ann_boxes, min_max

class SyncMNISTGenerator():
    def __init__(self, dataset_path,split, max_epochs = 100, num_classes=46, num_digits=0,
            worker_id=0, max_workers=1, dataset_fraction_used=1.0):
        no_videos_per_pickle = 1000
        self.num_classes = num_classes
        self.max_epochs = max_epochs
        self.train_files = glob(dataset_path + '/data*pickle')

        # if num_digits == 5:
        #     train_dataset_files = f'/data/datasets/video_mnist/files_order_pytorch/{split}_files_order3.pickle'
        # #elif num_digits == 3:
        # else:
        #     train_dataset_files = f'/data/datasets/video_mnist/files_order_pytorch/{split}_files_{num_digits}digits_order3.pickle'
        # if os.path.exists(train_dataset_files):
        #     with open(train_dataset_files, 'rb') as f:
        #         self.mnist_random_order = pickle.load(f)
        #     print(f'Reading from: {self.mnist_random_order[0]}')
        # else:
        #     print('Shuffling and saving train files')
        #     self.mnist_random_order = []
        #     for ep in range(100):
        #         np.random.shuffle(self.train_files)
        #         self.mnist_random_order.append(self.train_files)
        #     with open(train_dataset_files, 'wb') as f:
        #         pickle.dump(self.mnist_random_order, f)
        # print(f'Reading: {len(self.mnist_random_order[0])} pickles')
        
        overall_start = 0
        # print(f'dataset_path: {dataset_path}')
        # print(f'len(self.train_files): {len(self.train_files)}')
        # print(f'dataset_fraction_used: {dataset_fraction_used}')
        overall_end = int( dataset_fraction_used * len(self.train_files))
        
        per_worker = int(math.ceil((overall_end - overall_start) / float(max_workers)))
        self.start = overall_start + worker_id * per_worker
        self.end = min(self.start + per_worker, overall_end)
        # print(f'generator has overall_end:{overall_end} start-end {self.start}-{self.end}')

    def next_item(self, idx):
        #for epoch in range(self.max_epochs):
        #    print(f"Generator epoch: {epoch}")
        #return 1
        train_files = self.train_files #self.mnist_random_order[0]
        train_files = train_files[self.start:self.end]
        # print(f'Read:{train_files}')
        for file in train_files:
            train_videos, train_labels, target_boxes_np, is_ann_boxes = read_data_mnist(file, num_classes=self.num_classes)
            # train_videos, train_labels, target_boxes_np, is_ann_boxes, min_max_label = read_data_mnist(file, num_classes=self.num_classes)
            
            for pick_i in range(train_videos.shape[0]):
                # print(f'idx [{idx}] element {pick_i} from {file}')
                # video_ids = np.zeros_like(train_labels[pick_i], dtype=np.float32)
                #yield (train_videos[pick_i], train_labels[pick_i],video_ids, target_boxes_np[pick_i], is_ann_boxes[pick_i], pick_i, file)
                video = train_videos[pick_i] #.astype(np.float32) / 255.0 * 2 - 1.0
                yield video, train_labels[pick_i], target_boxes_np[pick_i]#, min_max_label[pick_i]


if __name__ == "__main__":
    if False:
        dataset_path = '/data/datasets/video_mnist/sync_mnist_large_v2_split_test_max_sync_dist_160_num_classes_46_no_digits_5_no_noise_parts_0/'
        gen = SyncMNISTGenerator(dataset_path=dataset_path, split='test')

        get_next_item = gen.next_item() 
        nr_videos = 15000

        time1 = time.time()
        # for b in range(nr_videos):
        for b, (train_videos, train_labels, target_boxes_np, _ , _) in enumerate(get_next_item):
            #    train_videos, train_labels, target_boxes_np, _ , _ = next(get_next_item)
            print(f'[{b}] {train_videos.mean()}')
        time2 = time.time()
        print(f'time read {nr_videos} videos: {time2 - time1}')
    
    else:
        batch_size = 8
        ds = SyncedMNISTDataSet(split='train')
        loader = torch.utils.data.DataLoader(ds,batch_size=batch_size, num_workers=2, worker_init_fn=syncMNIST_worker_init_fn)
        time1 = time.time()
        
        
        #for b, (train_videos, train_labels, target_boxes_np, _ , _, _, _) in enumerate(loader):
        for b, (train_videos, train_labels) in enumerate(loader):
            if b % 100 == 0:
                print(b * batch_size)
            pass
           # pdb.set_trace()
        #    print(f'[{b}] {train_videos.shape}, {train_labels.shape}')
        
        # print('Epoch 2')
        # for b, (train_videos, train_labels, target_boxes_np, _ , _) in enumerate(loader):
        #    # pdb.set_trace()
        #    print(f'[{b}] {train_videos[0].numpy().mean()}')

        

        time2 = time.time()
        print(f'time read : {time2 - time1}')
        
