# Code for "TSM: Temporal Shift Module for Efficient Video Understanding"
# arXiv:1811.08383
# Ji Lin*, Chuang Gan, Song Han
# {jilin, songhan}@mit.edu, ganchuang@csail.mit.edu

import os
import threading
import pdb

NUM_THREADS = 12
# VIDEO_ROOT = '/ssd/video/something/v2/20bn-something-something-v2'         # Downloaded webm videos
# FRAME_ROOT = '/ssd/video/something/v2/20bn-something-something-v2-frames'  # Directory for extracted frames
# VIDEO_ROOT = '/data/datasets/smt-smt-V2/20bn-something-something-v2'
# FRAME_ROOT = '/data/datasets/smt-smt-V2/20bn-something-something-v2-frames'

num_sec = 5
VIDEO_ROOT = '/data/datasets/in_the_wild/gifs'
FRAME_ROOT = f'/data/datasets/in_the_wild/gifs-frames-{num_sec}s'

# VIDEO_ROOT = '/data/datasets/in_the_wild/dataset_imar'
# FRAME_ROOT = f'/data/datasets/in_the_wild/dataset_imar-frames-{num_sec}s'


def split(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]


def extract(video, tmpl='%06d.jpg'):
    # os.system(f'ffmpeg -i {VIDEO_ROOT}/{video} -vf -threads 1 -vf scale=-1:256 -q:v 0 '
    #           f'{FRAME_ROOT}/{video[:-5]}/{tmpl}')
    # cmd0 = 'ffmpeg -i \"{}/{}\" -threads 1 -vf scale=-1:256 -q:v 0 \"{}/{}/%06d.jpg\"'.format(VIDEO_ROOT, video,
                                                                                            #  FRAME_ROOT, video[:-5])
    
    cmd = f'ffmpeg -t 00:0{num_sec} '
    cmd = cmd + '-i \"{}/{}\" -threads 1 -vf scale=-1:256 -q:v 0 \"{}/{}/%06d.jpg\"'.format(VIDEO_ROOT, video,
                                                                                             FRAME_ROOT, video[:-5])
   
    
    os.system(cmd)


def target(video_list):
    for video in video_list:
        video_path = os.path.join(FRAME_ROOT, video[:-5])
        if not os.path.exists(video_path):
            #print(f'video {video_path} does not exists')
            os.makedirs(os.path.join(FRAME_ROOT, video[:-5]))
            extract(video)
        else:
            dir_files = os.listdir(os.path.join(FRAME_ROOT, video[:-5]))
            if len(dir_files) <= 10:
                print(f'folder {video} has only {len(dir_files)} frames')
                extract(video)



if __name__ == '__main__':
    if not os.path.exists(VIDEO_ROOT):
        raise ValueError('Please download videos and set VIDEO_ROOT variable.')
    if not os.path.exists(FRAME_ROOT):
        os.makedirs(FRAME_ROOT)

    video_list = os.listdir(VIDEO_ROOT)
    splits = list(split(video_list, NUM_THREADS))

    threads = []
    for i, split in enumerate(splits):
        #target(split)
        thread = threading.Thread(target=target, args=(split,))
        thread.start()
        threads.append(thread)

    for thread in threads:
        thread.join()