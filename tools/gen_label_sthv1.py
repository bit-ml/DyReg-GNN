# Code for "TSM: Temporal Shift Module for Efficient Video Understanding"
# arXiv:1811.08383
# Ji Lin*, Chuang Gan, Song Han
# {jilin, songhan}@mit.edu, ganchuang@csail.mit.edu
# ------------------------------------------------------
# Code adapted from https://github.com/metalbubble/TRN-pytorch/blob/master/process_dataset.py
# processing the raw data of the video Something-Something-V1

import os

if __name__ == '__main__':
    dataset_name = '/data/datasets/something-something/something-something-v1'  # 'jester-v1''something-something-v1'  # 'jester-v1'
    with open('%s-labels.csv' % dataset_name) as f:
        lines = f.readlines()
    categories = []
    for line in lines:
        line = line.rstrip()
        categories.append(line)
    categories = sorted(categories)
    with open('smtv1_category.txt', 'w') as f:
        f.write('\n'.join(categories))

    dict_categories = {}
    for i, category in enumerate(categories):
        dict_categories[category] = i

    files_input = ['%s-validation.csv' % dataset_name, '%s-train.csv' % dataset_name]
    files_output = ['smtv1-val_videofolder.txt', 'smtv1-train_videofolder.txt']
    for (filename_input, filename_output) in zip(files_input, files_output):
        if 'val' in filename_output:
            split = 'valid'
        elif 'train' in filename_output:
            split = 'train'
        with open(filename_input) as f:
            lines = f.readlines()
        folders = []
        idx_categories = []
        for line in lines:
            line = line.rstrip()
            items = line.split(';')
            folders.append(items[0])
            idx_categories.append(dict_categories[items[1]])
        output = []
        for i in range(len(folders)):
            curFolder = folders[i]
            curIDX = idx_categories[i]
            # counting the number of frames in each video folders
            video_folder = os.path.join(f'/data/datasets/something-something/20bn-something-something-v1/{split}/', curFolder)
            
            # dir_files = os.listdir(os.path.join('../img', curFolder))
            # output.append('%s %d %d' % ('something/v1/img/' + curFolder, len(dir_files), curIDX))
            # print('%d/%d' % (i, len(folders)))
            if os.path.exists(video_folder):
                dir_files = os.listdir(video_folder)
                output.append('%s %d %d' % (curFolder, len(dir_files), curIDX))
                print('%d/%d' % (i, len(folders)))
            else:
                print(f'video {video_folder} does not exist: skipping')

        with open(filename_output, 'w') as f:
            f.write('\n'.join(output))
