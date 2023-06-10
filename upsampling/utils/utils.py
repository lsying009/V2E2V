import os
from pathlib import Path
from typing import Union, List
import pandas as pd
import numpy as np

from .const import fps_filename, timestamps_filename, imgs_dirname, img_formats, video_formats, frames_dirname, events_dirname
from .dataset import Sequence, ImageSequence, ImageSequenceTime, VideoSequence


def make_train_txt(data_dir: str, txt_name: str, num_intervals:int, step:int):
    txt_file = os.path.join(data_dir, txt_name)
    with open(txt_file, 'w') as f:
        print('Write training TXT ... \n')
        seq_names = os.listdir(data_dir) #[f for f in os.listdir(data_dir) if os.path.isdir(f)]
        seq_names.sort()
        video_idx = 0
        for seq_name in seq_names:
            if seq_name != 'slider_depth':
                continue

            path_to_seq = os.path.join(data_dir, seq_name)
            img_dirpath = os.path.join(path_to_seq, frames_dirname)
            
            
            if not os.path.exists(img_dirpath):
                continue
            event_dirpath = os.path.join(path_to_seq, events_dirname)
            if not os.path.exists(event_dirpath):
                continue
            timestamps_file = os.path.join(img_dirpath, 'timestamps.txt')
            timestamps = []
            with open(timestamps_file, 'r') as ft:   
                for line in ft.readlines():
                    idx, t = line.split()
                    timestamps.append(t)
            
            imgfile_names = [f for f in os.listdir(img_dirpath) if Path(f).suffix.lower() in img_formats]
            imgfile_names.sort()
            path_to_frames = [os.path.join(seq_name, frames_dirname, f) for f in imgfile_names]
 
            eventfile_names = [f for f in os.listdir(event_dirpath) if Path(f).suffix.lower() in ['.npz']]
            eventfile_names.sort()
            path_to_events = [os.path.join(seq_name, events_dirname, f) for f in eventfile_names]
            
            
            for frame_idx in range(0, len(path_to_frames)-num_intervals-1, step):
                # path_to_zip = os.path.join(path_to_cur_seq.split('/')[-1], '{}.zip'.format(path_to_cur_seq.split('/')[-1]))
                path_to_all_events = ' '.join( path_to_events[frame_idx+i] for i in range(num_intervals))
                path_to_all_frames = ' '.join( path_to_frames[frame_idx+i] for i in range(num_intervals+1))
                f.write(str(video_idx)+' '+ timestamps[frame_idx] +' '+ timestamps[frame_idx+num_intervals] +' '+ \
                    path_to_all_frames+' '+\
                    path_to_all_events+'\n')
            video_idx += 1
    print('Finished! \n')
    f.close()


def make_train_txt_wo_events(data_dir: str, txt_name: str, num_frames: int, step: int):
    txt_file = os.path.join(data_dir, txt_name)
    with open(txt_file, 'w') as f:
        print('Write training TXT ... \n')
        seq_names = os.listdir(data_dir) #[f for f in os.listdir(data_dir) if os.path.isdir(f)]
        seq_names.sort()
        video_idx = 0
        for seq_name in seq_names:
            path_to_seq = os.path.join(data_dir, seq_name)
            img_dirpath = os.path.join(path_to_seq, frames_dirname)
            
            # print(img_dirpath)
            if not os.path.exists(img_dirpath):
                continue

            timestamps_file = os.path.join(img_dirpath, 'timestamps.txt')
            timestamps = []
            with open(timestamps_file, 'r') as ft:   
                for line in ft.readlines():
                    idx, t = line.split()
                    timestamps.append(t)
            
            imgfile_names = [f for f in os.listdir(img_dirpath) if Path(f).suffix.lower() in img_formats]
            imgfile_names.sort()
            path_to_frames = [os.path.join(seq_name, frames_dirname, f) for f in imgfile_names]

            for frame_idx in range(0, len(path_to_frames)-num_frames+1, step):
                path_to_all_frames = ' '.join( path_to_frames[frame_idx+i] for i in range(num_frames))
                # path_to_zip = os.path.join(path_to_cur_seq.split('/')[-1], '{}.zip'.format(path_to_cur_seq.split('/')[-1]))
                f.write(str(video_idx)+' '+ timestamps[frame_idx] +' '+ timestamps[frame_idx+num_frames-1] +' '+ \
                    path_to_all_frames+'\n')
            video_idx += 1
    print('Finished! \n')
    f.close()


def is_video_file(filepath: str) -> bool:
    return Path(filepath).suffix.lower() in video_formats

def get_fps_file(dirpath: str) -> Union[None, str]:
    fps_file = os.path.join(dirpath, fps_filename)
    if os.path.isfile(fps_file):
        return fps_file
    return None

def get_timestamps_file(dirpath: str) -> Union[None, str]:
    timestamps_file = os.path.join(dirpath, timestamps_filename)
    if os.path.isfile(timestamps_file):
        return timestamps_file
    return None

def get_imgs_directory(dirpath: str) -> Union[None, str]:
    imgs_dir = os.path.join(dirpath) if imgs_dirname is None else os.path.join(dirpath, imgs_dirname)
    if os.path.isdir(imgs_dir):
        return imgs_dir
    return None

def get_video_file(dirpath: str) -> Union[None, str]:
    filenames = [f for f in os.listdir(dirpath) if is_video_file(f)]
    if len(filenames) == 0:
        return None
    assert len(filenames) == 1
    filepath = os.path.join(dirpath, filenames[0])
    return filepath

def fps_from_file(fps_file) -> float:
    assert os.path.isfile(fps_file)
    with open(fps_file, 'r') as f:
        fps = float(f.readline().strip())
    assert fps > 0, 'Expected fps to be larger than 0. Instead got fps={}'.format(fps)
    return fps

def timestamps_from_file(timestamps_file, data_mode='sim') -> List[float]:
    assert os.path.isfile(timestamps_file)
    if data_mode == 'sim':
        timestamps_reader = pd.read_csv(timestamps_file, 
                            iterator=False,
                            delimiter=' ',
                            names=['id', 't', 'start_id', 'end_id', 'num'],
                            dtype={'id':np.int16, 't': np.float32, 'start_id': np.int32, 'end_id': np.int32, 'num': np.int32},
                            engine='c',
                            index_col=False)
        # print(timestamps_reader)
        timestamps = timestamps_reader['t'].values ###########3？
    elif data_mode == 'real':
        timestamps_reader = pd.read_csv(timestamps_file, 
                            iterator=False,
                            delimiter=' ',
                            names=['t', 'img_path'],
                            dtype={'t': np.float64, 'img_path': str},
                            engine='c',
                            index_col=False)
        # print(timestamps_reader)
        timestamps = timestamps_reader['t'].values ###########3？

    return timestamps

#data_mode='sim'/real
def get_sequence_or_none(dirpath: str, len_seq:int=None, is_with_event:bool=False, data_mode:str='sim') -> Union[None, Sequence]:
    fps_file = get_fps_file(dirpath)
    if fps_file:
        # Must be a sequence (either ImageSequence or VideoSequence)
        fps = fps_from_file(fps_file)
        imgs_dir = get_imgs_directory(dirpath)
        if imgs_dir:
            return ImageSequence(imgs_dir, fps, len_seq)
        video_file = get_video_file(dirpath)
        assert video_file is not None
        return VideoSequence(video_file, fps, len_seq)

    timestamps_path = os.path.dirname(dirpath) if data_mode == 'real' else dirpath
    timestamps_file = get_timestamps_file(timestamps_path)
    if timestamps_file:
        timestamps = timestamps_from_file(timestamps_file, data_mode)
        imgs_dir = get_imgs_directory(dirpath)
        if imgs_dir:
            return ImageSequenceTime(imgs_dir, timestamps, len_seq, is_with_event, data_mode)
        # video_file = get_video_file(dirpath)
        # assert video_file is not None
        # return VideoSequence(video_file, fps)
    # Can be VideoSequence if there is a video file. But have to use fps from meta data.
    video_file = get_video_file(dirpath)
    if video_file is not None:
        return VideoSequence(video_file, len_seq=len_seq)
    return None


