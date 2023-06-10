import torch
import numpy as np
import cv2
import os

from upsampling.utils.upsamp_sequence import Upsampler
from utils.event_process import event_preprocess, events_to_voxel_grid
from .event_readers import SingleEventReaderNpz, RefTimeEventReaderZip


def read_timestamps_file(path_to_timestamps, unit='s'):
    ''' Read timestamps from a .txt file
        # TODO customise your according to the format of timestamps
        unit: str, 's' / 'us' / 'ms'
            Time unit of the time in the file, should rescaled to seconds [s]
        Output: List of float
            List timestamps in seconds

    '''
    timestamps = []
    if path_to_timestamps.split('/')[-1] == 'timestamps.txt':
        with open(path_to_timestamps, 'r') as f:
            for line in f:
                timestamps.append(float(line.strip().split()[1]))
        f.close()
    else:
        with open(path_to_timestamps, 'r') as f:
            for line in f:
                timestamps.append(float(line.strip().split()[0]))
                
        f.close()
    # t0 = timestamps[0]
    timestamps = np.array(timestamps)
    if unit in ['us']:
        timestamps /= 1e6
    elif unit in ['ns']:
        timestamps /= 1e9
                
    return list(timestamps)


## parent class for video reader
class VR:
    def __init__(self, image_dim, num_bins=5, is_with_events=False):
        ''' Video_reader
        Load video sequences in a frame pack for each reconstruction
        image_dim: [height, width]
        
        '''
        self.height, self.width = image_dim
        self.prev_ts_cache = np.zeros(1, dtype=np.float64)
        self.frame_id = 0
        self.num_frames = -1
        self.timestamps = []
        
        ## Params for events
        self.is_with_events = is_with_events
        self.num_bins = num_bins
        self.ending = False

    def update_frame(self):
        ## TODO redefine
        return np.zeros((self.height, self.width),dtype=np.uint8), 0
        
    def update_events(self):
        return
    
    def update_frame_pack(self, num_pack_frames):
        '''
        Load frames and timestamps for one reconstruction
        Inputs:
            num_pack_frames: int
                number of frames for event generation / reconstruction
        Outputs: 
            frame_pack: np.ndarray [num_frames, height, width]
                Images for event generation, num_frames = num_pack_frames if its the first frame in the sequence,
                else num_frames = num_pack_frames - 1
            gt_frame: np.ndarray [height, width]
                Ground truth image for the reconstruction (the last frame in a pack)
            timestamps: np.ndarray, float [num_frames]
                Timestamps corresponding to each frame

        '''
        start_frame_id = self.frame_id
        if start_frame_id != 0: 
            num_pack_frames -= 1

        num_pack_frames = min(num_pack_frames, self.num_frames-self.frame_id)
        
        frame_pack, timestamps = [], []
        
        # update frame and timestamp
        for _ in range(num_pack_frames):
            frame, t = self.update_frame()
            frame_pack.append(frame)
            timestamps.append(t)
        gt_frame = frame_pack[-1]
        
        frame_pack = np.stack(frame_pack, 0)
        if start_frame_id != 0:
            timestamps = np.concatenate((self.prev_ts_cache, np.stack(timestamps,0)),0)
        else:
            timestamps = np.stack(timestamps, 0)

        self.prev_ts_cache[0] = timestamps[-1]
        
        return frame_pack, gt_frame, timestamps
    
    def update_event_frame_pack(self, limit_num_events=-1, mode='upsampled'):
        '''
        Load GT frame and corresponding events for one reconstruction
        Inputs:
            num_pack_frames: int, default: -1
                number of frames for one reconstruction ()
            limit_num_events: int, default: -1
                limited number of events per reconstruction
            mode: str, 'real' or 'upsampled', default: upsampled
                'real': real data sequences, load events between two consecutive frames and split them by limit_num_events
                'upsampled': simulated data sequences with high frame rate, load events between several frames to reach limit_num_events
        Outputs: 
            event_windows: np.ndarray [N, 4]
                loaded events, the 4 channels (t,x,y,p)
            gt_frame: np.ndarray [height, width]
                Ground truth image for the reconstruction (the last frame in a pack)


        '''
        
        ## Skip the first frame
        if self.frame_id == 0:
            frame, _  = self.update_frame()
        
        if limit_num_events > 0 and mode == 'upsampled': 
            ## for simulated data 
            sum_num_events = 0
            event_pack = []
            while (sum_num_events < limit_num_events) and (self.frame_id<self.num_frames):
                gt_frame, _  = self.update_frame()
                events = self.update_events()
                if events is not None:
                    event_pack.append(events)
                    sum_num_events += len(events)
                if len(event_pack)>1:
                    event_window = np.concatenate(event_pack, 0)
                else:
                    event_window = event_pack[0]
        else:    
            gt_frame, _  = self.update_frame()
            event_window = self.update_events()
            if event_window is None:
                event_window = []
            
        if self.frame_id >= self.num_frames:
            self.ending = True

        self.num_events = len(event_window)

        event_windows = []
        if limit_num_events <= 0 or mode == 'upsampled':
            # the number of events ususally < limit_num_events
            event_window = events_to_voxel_grid(event_window, 
                                                num_bins=self.num_bins,
                                                width=self.width,
                                                height=self.height)
            event_window = event_preprocess(event_window, filter_hot_pixel=True)
            event_windows.append(event_window)
        else: # mode == 'real', or with limit_num_events, the number of events ususally exceeds limit_num_events
            num_evs = round(event_window.shape[0] / limit_num_events)
            if num_evs == 0:
                num_evs = 1
            event_window = np.array_split(event_window, num_evs, axis=0)
            
            for i in range(num_evs):
                evs = events_to_voxel_grid(event_window[i], 
                                num_bins=self.num_bins,
                                width=self.width,
                                height=self.height)
                evs = event_preprocess(evs, filter_hot_pixel=True)
                event_windows.append(evs)
                
        return event_windows, gt_frame

    

class VideoInterpolator(VR):
    def __init__(self, image_dim, num_bins=5, is_with_events=False, device='cuda:1', time_unit='s'):
        super(VideoInterpolator, self).__init__(image_dim, num_bins, is_with_events)
        ''' Load LFR frames required video interpolation, data_type='upsampling'
        '''
        self.time_unit = time_unit
        self.device = device
        
    def initialize(self, path_to_sequence, num_load_frames=-1):
        ''' Initialize / reset variables for a video sequence 
            Read all the frames and timestamps from the folder (path_to_sequence)
            num_load_frames: total number of frames to be loaded in the sequence
        '''
        self.frame_id = 0
        self.event_id = 0
        self.ending = False
        
        path_to_frames = []
        path_to_events = []
        
        for root, dirs, files in os.walk(path_to_sequence):
            for file_name in files:
                if file_name.split('.')[-1] in ['jpg','png']:
                    path_to_frames.append(os.path.join(root, file_name))
                elif file_name in ['timestamps.txt', 'images.txt', 'timestamp.txt']:#.split('.')[-1] in ['txt']:
                    path_to_timestamps = os.path.join(root, file_name)
                elif self.is_with_events and (file_name.split('.')[-1] in ['npz'] or file_name in ['events.txt', 'events.zip', 'events.csv']):
                    path_to_events.append(os.path.join(root, file_name))
          
        path_to_frames.sort()
        if num_load_frames > 0:
            path_to_frames = path_to_frames[:num_load_frames]
        
        
        timestamps = read_timestamps_file(path_to_timestamps, self.time_unit)
        if num_load_frames > 0:
            timestamps = timestamps[:num_load_frames]

        demo_image = cv2.imread(path_to_frames[0], cv2.IMREAD_GRAYSCALE)
        self.height = (demo_image.shape[0] // 2) * 2
        self.width = (demo_image.shape[1]//2) *2 
        
        self.prev_ts_cache = np.zeros(1, dtype=np.float64)

        frames = []
        for path_to_frame in path_to_frames:
            frame = cv2.imread(path_to_frame, cv2.IMREAD_GRAYSCALE)
            frames.append(frame[:self.height, :self.width])

        self.upsampler = Upsampler(device=self.device, image_dim=[self.height, self.width], is_train=False)
        self.frames, self.timestamps = self.upsampler.upsampling(frames, timestamps)
        self.num_frames = len(self.timestamps)
        
        
        ## if load events, define event loader
        if len(path_to_events)>1:
            path_to_events.sort()
            if num_load_frames > 0:
                path_to_events = path_to_events[:num_load_frames]
            self.event_window_iterator = SingleEventReaderNpz(path_to_events)
        elif len(path_to_events) == 1:
            path_to_events = path_to_events[0]
            self.event_window_iterator = RefTimeEventReaderZip(path_to_events, self.timestamps)
            
        
        
    def update_frame(self, frame_id=None):
        if frame_id is not None:
            self.frame_id = frame_id
        
        self.frame_id += 1

        return self.frames[self.frame_id-1], self.timestamps[self.frame_id-1]

    def update_events(self):
        try:
            event_window = next(self.event_window_iterator)
        except StopIteration:
            event_window = None
        self.event_id += 1
        return event_window



class VideoReader(VR):
    def __init__(self, image_dim, ds=[1/4,1/4]):
        super(VideoReader, self).__init__(image_dim)
        '''Read HFR video in video format'''
        self.ds = ds
        
    def initialize(self, path_to_video, num_load_frames=-1):
        ''' Initialize / reset variables for a video sequence 
            Read all the frames and timestamps from the video
            num_load_frames: total number of frames to be loaded in the sequence
        '''
        cap = cv2.VideoCapture(path_to_video)
    
        if (cap.isOpened()== False):
            assert "Error opening video stream or file"
                
        self.frames, self.timestamps = [], []
        frame_number = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        num_load_frames = frame_number if num_load_frames < 0 else num_load_frames
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = 0
        while(cap.isOpened()):
            frame_exists, frame = cap.read()
            if frame_exists:
                if frame_count > num_load_frames:
                    break
                # timestamps.append(cap.get(cv2.CAP_PROP_POS_MSEC)/1000)
                self.timestamps.append(float(frame_count)/fps)
                
                frame_count += 1
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                gray = cv2.resize(gray, dsize=(int(gray.shape[1]*self.ds[1]), int(gray.shape[0]*self.ds[0])))
                if frame.shape[0] > frame.shape[1]:
                    gray = gray.T
                self.frames.append(gray)
                
            else:
                break
        
        cap.release()
        self.num_frames = len(self.frames)

        self.prev_ts_cache.fill(0)
        self.frame_id = 0
        
    def update_frame(self, frame_id=None): 
        if frame_id is not None:
            self.frame_id = frame_id
        frame = self.frames[self.frame_id]
        timestamp = self.timestamps[self.frame_id]
        self.frame_id += 1
    
        return frame, timestamp
    
             

class ImageReader(VR):
    def __init__(self, image_dim, num_bins=5, is_with_events=False, time_unit='s'):
        super(ImageReader, self).__init__(image_dim, num_bins, is_with_events)
        '''Load HFR video in image format from a folder'''
        self.time_unit = time_unit
        
        
    def initialize(self, path_to_sequence, num_load_frames=-1):
        ''' Initialise / reset variables for a video sequence 
            Read all the frames and timestamps from the folder (path_to_sequence)
            num_load_frames: total number of frames to be loaded in the sequence
        '''
        self.frame_id = 0
        self.event_id = 0
        self.ending = False
        
        self.path_to_frames = []
        path_to_events = []
        
        for root, dirs, files in os.walk(path_to_sequence):
            for file_name in files:
                if file_name.split('.')[-1] in ['jpg','png']:
                    self.path_to_frames.append(os.path.join(root, file_name))
                elif file_name in ['timestamps.txt', 'images.txt', 'timestamp.txt']:
                    path_to_timestamps = os.path.join(root, file_name)
                elif self.is_with_events and (file_name.split('.')[-1] in ['npz'] or file_name in ['events.txt', 'events.zip', 'events.csv']):
                    path_to_events.append(os.path.join(root, file_name))
        self.path_to_frames.sort()
        
        self.timestamps = []
        self.timestamps = read_timestamps_file(path_to_timestamps, self.time_unit)
        if num_load_frames > 0:
            self.path_to_frames = self.path_to_frames[:num_load_frames]
            self.timestamps = self.timestamps[:num_load_frames]

        
        self.num_frames = len(self.path_to_frames)
        
        demo_image = cv2.imread(self.path_to_frames[0], cv2.IMREAD_GRAYSCALE)
        self.height = (demo_image.shape[0] // 2) * 2
        self.width = (demo_image.shape[1]//2) *2 
        
        self.prev_ts_cache = np.zeros(1, dtype=np.float64)
        
        ## if load events, define event loader
        if len(path_to_events)>1:
            path_to_events.sort()
            if num_load_frames > 0:
                path_to_events = path_to_events[:num_load_frames]
            self.event_window_iterator = SingleEventReaderNpz(path_to_events)
        elif len(path_to_events)==1:
            path_to_events = path_to_events[0]
            self.event_window_iterator = RefTimeEventReaderZip(path_to_events, self.timestamps)

        
    def update_frame(self, frame_id=None): 
        if frame_id is not None:
            self.frame_id = frame_id
        frame = cv2.imread(self.path_to_frames[self.frame_id], cv2.IMREAD_GRAYSCALE)
        frame = frame[:self.height, :self.width]
        timestamp = self.timestamps[self.frame_id]
        self.frame_id += 1
        
        return frame, timestamp

    def update_events(self):
        try:
            event_window = next(self.event_window_iterator)
        except StopIteration:
            event_window = None
        self.event_id += 1
        return event_window

            
