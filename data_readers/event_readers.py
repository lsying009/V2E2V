import pandas as pd
from os.path import splitext
import numpy as np


class RefTimeEventReaderZip:
    """
    Reads events from a '.txt', '.csv or .zip' file, and packages the events into
    non-overlapping event windows, according to the timestamp of reference intensity images.
    **Note**: This reader is much slower than the FixedSizeEventReader.
              The reason is that the latter can use Pandas' very efficient cunk-based reading scheme implemented in C.
    """

    def __init__(self, path_to_event_file, T_image):
        file_extension = splitext(path_to_event_file)[1]
        assert(file_extension in ['.txt', '.csv', '.zip'])

        self.iterator = pd.read_csv(path_to_event_file, 
                        iterator=False,
                        delimiter=' ',
                        names=['t', 'x', 'y', 'p'],
                        dtype={'t': np.float64, 'x': np.int16, 'y': np.int16, 'p': np.int16},
                        engine='c',
                        index_col=False)
        self.T_image = np.array(T_image) - T_image[0]
        self.len = len(T_image)-1
        timestamps = self.iterator.loc[:,['t']].values
        self.t0 = T_image[0] 
        timestamps -= T_image[0] 
        self.bound_index = []
        for t in self.T_image:
            idx = np.where(timestamps >= t)[0]
            if len(idx)==0:
                idx = len(timestamps)-1
            else:
                idx = idx[0]
            self.bound_index.append(idx)

        self.frame_id = 0 
        
    def __iter__(self):
        return self

    def __del__(self):
        return self

    def __next__(self):
        if self.frame_id >= self.len:
            raise StopIteration
        
        first_time_stamp = self.bound_index[self.frame_id]
        new_time_stamp = self.bound_index[self.frame_id+1]
        event_window = self.iterator.values[first_time_stamp:new_time_stamp]
        
        event_window[:,0] -= self.t0
        self.frame_id += 1
        return event_window


class SingleEventReaderNpz:
    """
    For simulated data sequences
    Reads events from a list of '.npz' file, read event window one by one into
    """

    def __init__(self, path_to_events):
        self.path_to_events = path_to_events
        self.len = len(self.path_to_events)
        self.frame_id = 0 
        
    def __iter__(self):
        return self

    def __del__(self):
        return self

    def __next__(self):
        if self.frame_id >= self.len:
            raise StopIteration
        
        event_window = np.load(self.path_to_events[self.frame_id])
        event_window = np.stack((event_window["t"], event_window["x"], event_window["y"],event_window["p"]), axis=1)
        self.frame_id += 1
        return event_window



