import os
import pandas as pd
from os.path import splitext
import numpy as np
import zipfile
import pandas as pd

class RefTimeEventReader:
    """
    Reads events from a '.txt' file, and packages the events into
    non-overlapping event windows, according to the timestamp of reference intensity images.
    **Note**: This reader is much slower than the FixedSizeEventReader.
              The reason is that the latter can use Pandas' very efficient cunk-based reading scheme implemented in C.
    """

    def __init__(self, path_to_event_file, T_image, start_stamp_index=0, start_index=0):
        file_extension = splitext(path_to_event_file)[1]
        assert(file_extension in ['.txt'])
        
        self.event_file = open(path_to_event_file, 'r')
        self.T_image = T_image

        # ignore header + lines before the first ref time stamp 
        self.start_index = start_index
        if self.start_index == 0:
            self.start_stamp = T_image[start_stamp_index]
            for i in range(100000000):
                line = self.event_file.readline()
                t, x, y, p = line.strip().split()
                if float(t) > self.start_stamp:
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



class FixedSizeEventReader:
    """
    Reads events from a '.txt' or '.zip' file, and packages the events into
    non-overlapping event windows, each containing a fixed number of events.
    """

    def __init__(self, path_to_event_file, num_events=10000, k_shift=-1, start_index=0):
        print('Will use fixed size event windows with {} events'.format(num_events))
        print('Output frame rate: variable')
        self.iterator = pd.read_csv(path_to_event_file, delim_whitespace=True, header=None,
                                    iterator=True,
                                    names=['t', 'x', 'y', 'pol'],
                                    dtype={'t': np.float64, 'x': np.int16, 'y': np.int16, 'pol': np.int16},
                                    engine='c', index_col=False,
                                    skiprows=start_index + 1, nrows=None, memory_map=True)
        self.num_events = num_events
        self.k_shift = k_shift
        self.prev_events_size = num_events - k_shift
        self.frame_idx = 0
    def __iter__(self):
        return self

    def __next__(self):
        if self.k_shift > 0:
            if self.frame_idx == 0:
                event_list = self.iterator.get_chunk(self.num_events)
                event_window = np.array(event_list)
                # self.prev_events = event_window[-self.prev_events_size:].copy()
            else:
                event_list = self.iterator.get_chunk(self.k_shift)
                event_window = np.array(event_list)
                event_window = np.concatenate((self.prev_events, event_window), 0)
            self.prev_events = event_window[-self.prev_events_size:].copy()
            # print(event_window.shape, event_window[0], event_window[-1])
                # print(self.prev_events.shape, event_window[0])
            self.frame_idx += 1
        else:
            event_list = self.iterator.get_chunk(self.num_events)
            event_window = np.array(event_list)

            # event_window = self.iterator.__next__().values
        return event_window


class FixedDurationEventReader:
    """
    Reads events from a '.txt' file, and packages the events into
    non-overlapping event windows, each of a fixed duration.
    **Note**: This reader is much slower than the FixedSizeEventReader.
              The reason is that the latter can use Pandas' very efficient cunk-based reading scheme implemented in C.
    """

    def __init__(self, path_to_event_file, duration_ms=50.0, start_index=0):
        print('Will use fixed duration event windows of size {:.2f} ms'.format(duration_ms))
        print('Output frame rate: {:.1f} Hz'.format(1000.0 / duration_ms))
        file_extension = splitext(path_to_event_file)[1]
        assert(file_extension in ['.txt'])
        
        self.event_file = open(path_to_event_file, 'r')

        # ignore header + the first start_index lines
        for i in range(1 + start_index):
            self.event_file.readline()

        self.last_stamp = None
        self.duration_s = duration_ms / 1000.0

    def __iter__(self):
        return self

    def __del__(self):
        self.event_file.close()

    def __next__(self):
        event_list = []
        for line in self.event_file:
            t, x, y, p = line.split(' ')
            t, x, y, p = float(t), int(x), int(y), int(p)
            event_list.append([t, x, y, p])
            if self.last_stamp is None:
                self.last_stamp = t
            if t > self.last_stamp + self.duration_s:
                self.last_stamp = t
                event_window = np.array(event_list)
                return event_window

        raise StopIteration


class RefTimeEventReaderCSV:
    # csv file in zip, [t, x, y, p]
    def __init__(self, path_to_event_file, start_idx, num_event_list):
        self.z = zipfile.ZipFile(path_to_event_file) 
        self.f_csv = self.z.open('events.csv')
        self.num_event_list = num_event_list
        self.len = len(num_event_list)
        
        self.iterator = pd.read_csv(self.f_csv, 
                        iterator=True,
                        delimiter=' ',
                        names=['t', 'x', 'y', 'p'],
                        dtype={'t': np.float64, 'x': np.int16, 'y': np.int16, 'p': np.int16},
                        engine='c',
                        index_col=False)
        # read start
        self.iterator.get_chunk(start_idx)
        self.frame_id = 0

    def __iter__(self):
        return self

    def __del__(self):
        self.f_csv.close()
        self.z.close()
    def __next__(self):
        if self.frame_id >= self.len:
            raise StopIteration

        event_list = self.iterator.get_chunk(self.num_event_list[self.frame_id])
        event_window = np.array(event_list)
        self.frame_id += 1
            
        return event_window
                

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
        self.t0 = T_image[0] #np.copy(timestamps[0])
        timestamps -= T_image[0] #timestamps[0]
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
    For simulated datasets
    Reads events from a list of '.npz' file, corresponding to images, read event window one by one into
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



