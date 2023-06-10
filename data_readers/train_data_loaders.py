import numpy as np
import os
import torch
import torchvision.transforms as transforms
import torch.utils.data as data 
import cv2
from utils.event_process import *


class TrainSeqData(data.Dataset):
    def __init__(self, train_data_txt, path_to_train_data, len_sequence, num_pack_frames):
        '''Training sequence loader
           Load data sequence to train v2e2v model for many-to-one strategy
           The loss is computed once after a sequence of reconstructions with length <len_sequence>
        
           The format of train_data_txt: 
           seq_id [timestamps (in seconds) of frames with a length of <in_channels>] [path_to_frames (upsampled) with a length of <in_channels>]
           These upsampled frames are used to generate events and finally reconstruct one frame
           The ground truth image is the last frame in each line
           The total number of frames in each sequence: (in_channels-1) x len_sequence + 1
        '''
        
        self.txt_file = train_data_txt
        self.path_to_train_data = path_to_train_data

        self.len_sequence = len_sequence # number of reconstructions in each sequence
        self.num_pack_frames = num_pack_frames # number of frames for each reconstruction 

        self.to_tensor = transforms.ToTensor()

        self.video_cnt = []
        self.image_paths = []
        self.timestamps = []
       
        cur_line_cnt = []
        prev_video_cnt = 0
        line_id = 0
        with open(self.txt_file,'rb') as f:
            for line in f:
                str_list = line.strip().split()
                
                v_cnt = int(str_list[0]) # v_cnt: video/seq ID
                if v_cnt != prev_video_cnt:
                    self.video_cnt.append(cur_line_cnt) #video_cnt
                    cur_line_cnt = []
                    prev_video_cnt = v_cnt
                cur_line_cnt.append(line_id)
                line_id += 1
                
                for i in range(self.num_pack_frames):
                    self.timestamps.append(float(str_list[1+i]))
                    self.image_paths.append(os.path.join(self.path_to_train_data, str(str_list[self.num_pack_frames+1+i], encoding = "utf-8"))) #cur_img_path1
            self.video_cnt.append(cur_line_cnt) #video_cnt
        f.close()
        self._divide_seq_info()

    def _divide_seq_info(self):
        '''Divide training data into sequences'''
        self.start_seq_id = []
        self.len_seq = []
        step = 5 #self.num_pack_frames if self.use_upsamp_data else 5

        # print('video_num:', len(self.video_cnt))
        for line_id_in_video in self.video_cnt:
            length_video = len(line_id_in_video)
            # print('length_video:',length_video)
            for idx in range(0, length_video, step):
                if idx + self.len_sequence <= length_video:
                    # print('line_id, len_seq:', line_id_in_video[idx], self.len_sequence)
                    self.start_seq_id.append(line_id_in_video[idx])
                    self.len_seq.append(self.len_sequence)
                elif length_video-idx>=3:
                    # print('line_id, len_seq:', line_id_in_video[idx], length_video-idx)
                    self.start_seq_id.append(line_id_in_video[idx])
                    self.len_seq.append(length_video-idx)
    
    def __len__(self):
        return len(self.start_seq_id)
    
    def __getitem__(self, index):
        # return sequence of timestamps, images, and ground truth images for each reconstruction
        seq_id = self.start_seq_id[index]
        cur_len_seq = self.len_seq[index]

        seq_timestamps, seq_images, seq_gt_images = [], [], []
        for m in range(cur_len_seq):
            start_id = (seq_id+m)*self.num_pack_frames
            timestamps = self.timestamps[start_id:start_id+self.num_pack_frames]
            

            imgs = []

            for i in range(self.num_pack_frames):
                img = np.float32((cv2.imread(self.image_paths[(seq_id+m)*self.num_pack_frames+i], cv2.IMREAD_GRAYSCALE))) # /255.0
                imgs.append(img)
            
            images = self.to_tensor(np.stack(imgs, 2))
            timestamps = torch.from_numpy(np.array(timestamps))
            seq_timestamps.append(timestamps)
            seq_images.append(images)
            seq_gt_images.append(images[-1:]/255.0)

        return seq_timestamps, seq_images, seq_gt_images


class TrainFixNEventData(data.Dataset):
    '''Training sequence loader
        Load data sequence to train E2V reconstruction model for many-to-one strategy
        The loss is computed once after a sequence of reconstructions with length of <len_sequence>
        The number of events per reconstruction is limited to ~<limit_num_events> (15000 as default)
        
        The format of train_data_txt: 
        seq_id num_events timestamp_0(in seconds) timestamp_1 path_to_frame_0 path_to_frame_1 path_to_events between frame_0 and frame_1

    '''
    def __init__(self, train_data_txt, cfgs):
        self.txt_file = train_data_txt
        self.path_to_train_data = cfgs.path_to_train_data
        self.num_bins = cfgs.num_bins
        self.height, self.width = cfgs.image_dim
        self.limit_num_events = cfgs.num_events
        self.len_sequence = cfgs.len_sequence
        self.add_noise = cfgs.add_noise
        
        self.video_cnt = []
        self.event_paths = []
        self.image_paths = []
        self.next_image_paths = []
        self.num_events_list = []
        
        self.to_tensor = transforms.ToTensor()

        with open(self.txt_file,'rb') as f:
            for line in f:
                str_list = line.strip().split()
                self.video_cnt.append(int(str_list[0])) #video_cnt
                self.num_events_list.append(int(str_list[1]))
                self.image_paths.append(str(str_list[4], encoding = "utf-8")) #cur_img_path1
                self.next_image_paths.append(str(str_list[5], encoding = "utf-8")) #cur_next_img_path
                self.event_paths.append(str(str_list[6], encoding = "utf-8"))
        f.close()
        
        self.split_sequences()
        
    
    def __len__(self):
        return len(self.sequence_line_id)   
    
    def split_sequences(self):
        prev_video_id = -1
        sum_num_events = 0
        self.sequence_line_id = []
        line_id_per_reconstruction = []
        line_id_per_sequence = []
        frame_cnt, single_frame_cnt = 0, 0
        for line_id, video_id in enumerate(self.video_cnt):
            if video_id != prev_video_id:
                if len(line_id_per_sequence)>=5:
                    if line_id_per_reconstruction:
                        line_id_per_sequence.append(line_id_per_reconstruction)
                    self.sequence_line_id.append(line_id_per_sequence)
                line_id_per_sequence = []
                line_id_per_reconstruction = []
                prev_video_id = video_id
                sum_num_events = 0
                single_frame_cnt = 0
                frame_cnt = 0
                
            cur_num_event = self.num_events_list[line_id]
            sum_num_events += cur_num_event
            line_id_per_reconstruction.append(line_id)
            single_frame_cnt += 1
            if sum_num_events >= self.limit_num_events or (single_frame_cnt==1 and sum_num_events > 0.8*self.limit_num_events):
                line_id_per_sequence.append(line_id_per_reconstruction)
                frame_cnt += 1
                sum_num_events = 0
                single_frame_cnt = 0
                line_id_per_reconstruction = []
                
            if frame_cnt >= self.len_sequence:
                self.sequence_line_id.append(line_id_per_sequence)
                line_id_per_sequence = []
                line_id_per_reconstruction = []
                frame_cnt = 0
                
   
    def _e2_voxelgrid(self, event_patch):
        event_patch = events_to_voxel_grid(event_patch, 
                                            num_bins=self.num_bins,
                                            width=self.width,
                                            height=self.height)
        event_patch = event_preprocess(event_patch, filter_hot_pixel=False)
        event_patch = torch.from_numpy(event_patch)
        return event_patch
    
    
    def __getitem__(self, index):
        line_id_per_sequence = self.sequence_line_id[index]

        seq_events = []
        for line_id_per_reconstruction in line_id_per_sequence:
            event_window = np.empty((0,4),dtype=np.float32)
            for line_id in line_id_per_reconstruction:
                event_path = os.path.join(self.path_to_train_data, self.event_paths[line_id])
                cur_event_window = np.load(event_path, allow_pickle=True) #["arr_0"]
                cur_event_window = np.stack((cur_event_window["t"], cur_event_window["x"], cur_event_window["y"],cur_event_window["p"]), axis=1)
                event_window = np.concatenate((event_window, cur_event_window), 0)
                
            event_patch = self._e2_voxelgrid(event_window) 
            if self.add_noise:
                event_patch = add_noise_to_voxel(event_patch, noise_std=0.1, noise_fraction=1)

            seq_events.append(event_patch)
        
        img = np.float32((cv2.imread(os.path.join(self.path_to_train_data, self.image_paths[line_id_per_sequence[0][0]]), cv2.IMREAD_GRAYSCALE))/255.0)   
        gt_img = np.float32((cv2.imread(os.path.join(self.path_to_train_data, self.next_image_paths[line_id_per_sequence[-1][-1]]), cv2.IMREAD_GRAYSCALE))/255.0)

        img = self.to_tensor(img)
        gt_img = self.to_tensor(gt_img)
        

        return seq_events, img, gt_img

