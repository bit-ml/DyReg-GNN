# Code for "TSM: Temporal Shift Module for Efficient Video Understanding"
# arXiv:1811.08383
# Ji Lin*, Chuang Gan, Song Han
# {jilin, songhan}@mit.edu, ganchuang@csail.mit.edu
# ------------------------------------------------------
# Code adapted from https://github.com/metalbubble/TRN-pytorch/blob/master/process_dataset.py
# processing the raw data of the video Something-Something-V2

import os
import json
DATA_UTILS_ROOT='./data/smt-smt-V2/tsm_data/'
FRAME_ROOT='./data/smt-smt-V2/smt-smt-V2-frames/'

if __name__ == '__main__':
    dataset_name = DATA_UTILS_ROOT+'something-something-v2'  
    with open('%s-labels.json' % dataset_name) as f:
        data = json.load(f)
    categories = []
    for i, (cat, idx) in enumerate(data.items()):
        assert i == int(idx)  # make sure the rank is right
        categories.append(cat)

    with open(DATA_UTILS_ROOT+'category.txt', 'w') as f:
        f.write('\n'.join(categories))

    dict_categories = {}
    for i, category in enumerate(categories):
        dict_categories[category] = i

    files_input = ['%s-validation.json' % dataset_name, '%s-train.json' % dataset_name, '%s-test.json' % dataset_name]
    files_output = ['val_videofolder.txt', 'train_videofolder.txt', 'test_videofolder.txt']
    for (filename_input, filename_output) in zip(files_input, files_output):
        with open(filename_input) as f:
            data = json.load(f)
        folders = []
        idx_categories = []
        for item in data:
            folders.append(item['id'])
            if 'test' not in filename_input:
                idx_categories.append(dict_categories[item['template'].replace('[', '').replace(']', '')])
            else:
                idx_categories.append(0)
        output = []
        for i in range(len(folders)):
            curFolder = folders[i]
            curIDX = idx_categories[i]
            # counting the number of frames in each video folders
            video_folder = os.path.join(FRAME_ROOT, curFolder)
            if os.path.exists(video_folder):
                dir_files = os.listdir(video_folder)
                output.append('%s %d %d' % (curFolder, len(dir_files), curIDX))
                print('%d/%d' % (i, len(folders)))
            else:
                print(f'video {video_folder} does not exist: skipping')

        with open(DATA_UTILS_ROOT+filename_output, 'w') as f:
            f.write('\n'.join(output))
