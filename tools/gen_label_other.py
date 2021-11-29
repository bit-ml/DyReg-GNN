# Code for "TSM: Temporal Shift Module for Efficient Video Understanding"
# arXiv:1811.08383
# Ji Lin*, Chuang Gan, Song Han
# {jilin, songhan}@mit.edu, ganchuang@csail.mit.edu
# ------------------------------------------------------
# Code adapted from https://github.com/metalbubble/TRN-pytorch/blob/master/process_dataset.py
# processing the raw data of the video Something-Something-V2

import os
import json
import pdb

if __name__ == '__main__':
    num_sec = 5

    dataset_folder = f'/data/datasets/in_the_wild/gifs-frames-{num_sec}s' 
    filename_output = f'/data/datasets/in_the_wild/gifs-frames-{num_sec}s.txt'

    folders = os.listdir(dataset_folder)
    output = []
    for i in range(len(folders)):
        curFolder = folders[i]
        curIDX = 0
        # counting the number of frames in each video folders
        video_folder = os.path.join(dataset_folder, curFolder)
        if os.path.exists(video_folder):
            dir_files = os.listdir(video_folder)
            output.append('%s %d %d' % (curFolder, len(dir_files), curIDX))
            print('%d/%d' % (i, len(folders)))
        else:
            print(f'video {video_folder} does not exist: skipping')

    with open(filename_output, 'w') as f:
        f.write('\n'.join(output))
