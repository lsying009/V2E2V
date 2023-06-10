import os
from pathlib import Path
from typing import Union, List

import numpy as np
from fractions import Fraction
from PIL import Image
import skvideo.io
import torchvision.transforms as transforms

from .const import mean, std, img_formats

from math import sqrt, ceil, floor
from torch.nn import ReflectionPad2d
import numpy as np
import torch

def optimal_crop_size(max_size, max_subsample_factor):
    """ Find the optimal crop size for a given max_size and subsample_factor.
        The optimal crop size is the smallest integer which is greater or equal than max_size,
        while being divisible by 2^max_subsample_factor.
    """
    crop_size = int(pow(2, max_subsample_factor) * ceil(max_size / pow(2, max_subsample_factor)))
    return crop_size

class CropParameters:
    """ Helper class to compute and store useful parameters for pre-processing and post-processing
        of images in and out of E2VID.
        Pre-processing: finding the best image size for the network, and padding the input image with zeros
        Post-processing: Crop the output image back to the original image size
    """

    def __init__(self, width, height, num_encoders):

        self.height = height
        self.width = width
        self.num_encoders = num_encoders
        self.width_crop_size = optimal_crop_size(self.width, num_encoders)
        self.height_crop_size = optimal_crop_size(self.height, num_encoders)

        self.padding_top = ceil(0.5 * (self.height_crop_size - self.height))
        self.padding_bottom = floor(0.5 * (self.height_crop_size - self.height))
        self.padding_left = ceil(0.5 * (self.width_crop_size - self.width))
        self.padding_right = floor(0.5 * (self.width_crop_size - self.width))
        self.pad = ReflectionPad2d((self.padding_left, self.padding_right, self.padding_top, self.padding_bottom))

        self.cx = floor(self.width_crop_size / 2)
        self.cy = floor(self.height_crop_size / 2)

        self.ix0 = self.cx - floor(self.width / 2)
        self.ix1 = self.cx + ceil(self.width / 2)
        self.iy0 = self.cy - floor(self.height / 2)
        self.iy1 = self.cy + ceil(self.height / 2)


class RefTimeEventReader:
    """
    Reads events from a '.txt' file, and packages the events into
    non-overlapping event windows, according to the timestamp of reference intensity images.
    **Note**: This reader is much slower than the FixedSizeEventReader.
              The reason is that the latter can use Pandas' very efficient cunk-based reading scheme implemented in C.
    """

    def __init__(self, path_to_event_file, T_image, start_stamp_index=0, start_index=0):
        file_extension = os.path.splitext(path_to_event_file)[1]
        assert(file_extension in ['.txt'])
        
        self.event_file = open(path_to_event_file, 'r')
        self.T_image = T_image

        # ignore header + lines before the first ref time stamp 
        self.start_index = start_index
        self.start_stamp = T_image[start_stamp_index]
        
        # self.reset_stamp = self.start_stamp if self.start_stamp > 100 else 0.
        # T_image = T_image - self.reset_stamp

        if self.start_index == 0:
            for i in range(100000000):
                line = self.event_file.readline()
                t, x, y, p = line.strip().split()
                if float(t) > self.start_stamp: #-self.reset_stamp
                    break
            self.start_index += i
        else:
            # ignore header + the first start_index lines
            for i in range(start_index):
                self.event_file.readline()

        self.start_stamp_index = start_stamp_index 
        # self.duration_s = T_image[self.start_stamp_index+1]- T_image[self.start_stamp_index]
        self.last_stamp = None
        self.stop_stamp = self.T_image[-1]
        
    def __iter__(self):
        return self

    def __del__(self):
        self.event_file.close()

    def __next__(self):
        if self.start_stamp_index != self.T_image.shape[0]-1:
            self.last_stamp = self.T_image[self.start_stamp_index+1]
        self.start_stamp_index += 1
            # read event txt
        event_list = []
        for idx, line in enumerate(self.event_file):
            t, x, y, p = line.strip().split()
            t, x, y, p = float(t), int(x), int(y), int(p)
            event_list.append([t, x, y, p])
            if self.start_stamp_index == self.T_image.shape[0]:
                raise StopIteration
            if t > self.last_stamp: 
                event_window = np.array(event_list)
                # end_index = self.start_index + idx
                return event_window #, end_index    

        raise StopIteration



class Sequence:
    def __init__(self):
        normalize = transforms.Normalize(mean=mean, std=std)
        self.transform = transforms.Compose([transforms.ToTensor(), normalize])
        

    def __iter__(self):
        return self

    def __next__(self):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError


class ImageSequence(Sequence):
    def __init__(self, imgs_dirpath: str, fps: float, len_seq: int=None):
        super().__init__()
        self.fps = fps

        assert os.path.isdir(imgs_dirpath)
        self.imgs_dirpath = imgs_dirpath

        self.file_names = [f for f in os.listdir(imgs_dirpath) if self._is_img_file(f)]
        assert self.file_names
        self.file_names.sort()
        self.len_seq = len_seq if len_seq is not None else len(self.file_names)

    @classmethod
    def _is_img_file(cls, path: str):
        return Path(path).suffix.lower() in img_formats

    def __next__(self):
        for idx in range(0, self.len_seq-1): #len(self.file_names) - 1
            file_paths = self._get_path_from_name([self.file_names[idx], self.file_names[idx + 1]])
            imgs = list()
            for file_path in file_paths:
                img = self._pil_loader(file_path)
                img = self.transform(img)
                imgs.append(img)
            times_sec = [idx/self.fps, (idx + 1)/self.fps]
            yield imgs, times_sec

    def __len__(self):
        return self.len_seq - 1

    @staticmethod
    def _pil_loader(path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            img = img.convert('RGB')
            

            w_orig, h_orig = img.size
            
            w, h = w_orig//32*32, h_orig//32*32

            left = (w_orig - w)//2
            upper = (h_orig - h)//2
            right = left + w
            lower = upper + h
            img = img.crop((left, upper, right, lower))
            return img

    def _get_path_from_name(self, file_names: Union[list, str]) -> Union[list, str]:
        if isinstance(file_names, list):
            return [os.path.join(self.imgs_dirpath, f) for f in file_names]
        return os.path.join(self.imgs_dirpath, file_names)


class ImageSequenceTime(Sequence):
    def __init__(self, imgs_dirpath: str, timestamps: List[float], len_seq:int=None, is_with_event:bool=False, data_mode:str='sim'):
        super().__init__()
        self.timestamps = timestamps
        # self.crop = CropParameters(image_dim[1], image_dim[0], 5)
        assert os.path.isdir(imgs_dirpath)
        self.imgs_dirpath = imgs_dirpath

        self.file_names = [f for f in os.listdir(imgs_dirpath) if self._is_img_file(f)]
        assert self.file_names
        self.file_names.sort()
        
        self.len_seq = len_seq if len_seq is not None else len(self.file_names)
        if self.imgs_dirpath.split('/')[-2] in ['office_zigzag']:
            self.len_seq = min(len_seq, 246)

        self.is_with_event = is_with_event
        self.data_mode = data_mode
        if self.is_with_event:
            if self.data_mode == 'sim':
                self.events_dirpath = os.path.join(os.path.dirname(imgs_dirpath), 'events')
                assert self.events_dirpath
                self.event_file_names = [f for f in os.listdir(self.events_dirpath) if Path(f).suffix.lower()=='.npz']
                self.event_file_names.sort()
            else: #'sequence'
                event_file = os.path.join(os.path.dirname(imgs_dirpath), 'events.txt')
                self.event_window_iterator = RefTimeEventReader(event_file, self.timestamps)
                # self.timestamps = self.timestamps - self.timestamps[0]


    @classmethod
    def _is_img_file(cls, path: str):
        return Path(path).suffix.lower() in img_formats

    def __next__(self):
        for idx in range(0, self.len_seq - 1):
            file_paths = self._get_path_from_name([self.file_names[idx], self.file_names[idx + 1]])
            imgs = list()
            for file_path in file_paths:
                img = self._pil_loader(file_path)
                img = self.transform(img)
                # print(img.shape)
                # img = self.crop.pad(img)
                imgs.append(img)
            times_sec = [self.timestamps[idx], self.timestamps[idx + 1]]
            if self.is_with_event:
                if self.data_mode == 'sim':
                    event_file_path =  os.path.join(self.events_dirpath, self.event_file_names[idx])
                    events = np.load(event_file_path) #x["arr_0"]
                    events = events["arr_0"]
                else:
                    events = self.event_window_iterator.__next__()
                yield imgs, times_sec, events
            else:
                yield imgs, times_sec

    def __len__(self):
        return self.len_seq - 1

    # @staticmethod
    def _pil_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            img = img.convert('RGB')

            # img = self.crop.pad(img)
            # w_orig, h_orig = img.size
            # w, h = w_orig//32*32, h_orig//32*32

            # left = (w_orig - w)//2
            # upper = (h_orig - h)//2
            # right = left + w
            # lower = upper + h
            # img = img.crop((left, upper, right, lower))
            return img

    def _get_path_from_name(self, file_names: Union[list, str]) -> Union[list, str]:
        if isinstance(file_names, list):
            return [os.path.join(self.imgs_dirpath, f) for f in file_names]
        return os.path.join(self.imgs_dirpath, file_names)


class VideoSequence(Sequence):
    def __init__(self, video_filepath: str, fps: float=None, len_seq:int=None):
        super().__init__()
        metadata = skvideo.io.ffprobe(video_filepath)
        self.fps = fps
        if self.fps is None:
            self.fps = float(Fraction(metadata['video']['@avg_frame_rate']))
            assert self.fps > 0, 'Could not retrieve fps from video metadata. fps: {}'.format(self.fps)
            print('Using video metadata: Got fps of {} frames/sec'.format(self.fps))

        # Length is number of frames - 1 (because we return pairs).
        self.len = int(metadata['video']['@nb_frames']) - 1 if len_seq is None else len_seq-1
        self.videogen = skvideo.io.vreader(video_filepath)
        self.last_frame = None

    def __next__(self):
        for idx, frame in enumerate(self.videogen):
            if idx > self.len:
                raise StopIteration
            h_orig, w_orig, _ = frame.shape
            w, h = w_orig//32*32, h_orig//32*32

            left = (w_orig - w)//2
            upper = (h_orig - h)//2
            right = left + w
            lower = upper + h
            frame = frame[upper:lower, left:right]
            assert frame.shape[:2] == (h, w)
            frame = self.transform(frame)

            if self.last_frame is None:
                self.last_frame = frame
                continue
            last_frame_copy = self.last_frame.detach().clone()
            self.last_frame = frame
            imgs = [last_frame_copy, frame]
            times_sec = [(idx - 1)/self.fps, idx/self.fps]
            yield imgs, times_sec

    def __len__(self):
        return self.len
