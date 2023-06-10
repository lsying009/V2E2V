import os
import GPUtil
# Get a list of available GPUs
gpus = GPUtil.getGPUs()
# Select the GPU with the lowest utilization
chosen_gpu = None
for gpu in gpus:
    if not chosen_gpu:
        chosen_gpu = gpu
    elif gpu.memoryUtil < chosen_gpu.memoryUtil:
        chosen_gpu = gpu
# Set CUDA device to the selected GPU
os.environ["CUDA_VISIBLE_DEVICES"] = str(chosen_gpu.id)


import torch
import torch.nn as nn
import numpy as np
import argparse
import cv2


from model_v2e2v import V2E2VNet
from data_readers.video_readers import VideoInterpolator, VideoReader, ImageReader
from utils.configs import set_configs
from utils.data_io import make_event_preview, ImageWriter, EventWriter
# from utils.evaluate import mse, psnr, ssim, PerceptualLoss
# from utils.image_process import normalize_image

class V2E2V(nn.Module):
    def __init__(self, cfgs, device):
        super(V2E2V, self).__init__()
        ''' V2E2V framework for inference: 
            v2e2v_net takes intenstiy video sequences as input and outputs reconstructed images
            which includes V2E generation with diversity in parameters,
            and E2V reconstruction network CISTA-LSTC
        '''
        self.image_dim = cfgs.image_dim # [H, W]
        self.reader_type = cfgs.reader_type # image_reader, upsampling, video
        self.num_pack_frames = cfgs.num_pack_frames # number of frames for each reconstruction
        self.device = device
        self.num_load_frames = cfgs.test_img_num # loaded number of frames for each sequence
        self.test_data_name = cfgs.test_data_name # test data_name
        self.time_unit = cfgs.time_unit
        
        # Load intensity frames
        if self.reader_type == 'video':
            # Load high frame rate video directly (adobe240 or other videos)
            self.path_to_sequences = []
            for file_name in os.listdir(cfgs.path_to_test_data):
                video_file = os.path.join(cfgs.path_to_test_data, file_name)
                if not os.path.isfile(video_file) or file_name.startswith('.') or file_name.split('.')[-1]=='txt':
                    continue
                self.path_to_sequences.append(video_file)
            self.path_to_sequences.sort()
            self.video_renderer = VideoReader(self.image_dim, ds=[1/4,1/4])
        else: # 'upsampling' or 'image_reader'
            # Load image sequences from a list of folder
            self.path_to_sequences = []
            for folder_name in os.listdir(cfgs.path_to_test_data):
                if os.path.isdir(os.path.join(cfgs.path_to_test_data, folder_name)):
                    self.path_to_sequences.append(os.path.join(cfgs.path_to_test_data, folder_name))
            self.path_to_sequences.sort()
            if self.reader_type == 'upsampling':
            # reader_type='upsampling': low frame rate (LFR) image sequences requires VideoInterpolator 
                self.video_renderer = VideoInterpolator(self.image_dim, device=self.device, time_unit=self.time_unit)
            else:
            # reader_type='image_reader': load HFR image sequences directly
                self.video_renderer = ImageReader(self.image_dim, time_unit=self.time_unit)
        

        # Load checkpoint of E2V reconstruction network
        self.model_name = os.path.splitext(cfgs.path_to_test_model.split('/')[-1])[0]
        checkpoint = torch.load(cfgs.path_to_test_model, map_location='cuda:0')

        # Initialise params of V2E model
        if 'v2e_params' in checkpoint:
            cfgs.C = checkpoint['v2e_params']['C']
            cfgs.ps = checkpoint['v2e_params']['ps']
            cfgs.pl = checkpoint['v2e_params']['pl']
            cfgs.cutoff_hz = checkpoint['v2e_params']['cutoff_hz']
            cfgs.qs = checkpoint['v2e_params']['qs']
            cfgs.ql = checkpoint['v2e_params']['ql']
            cfgs.refractory_period_s = checkpoint['v2e_params']['refractory_period_s']
        
        self.v2e2v_net = V2E2VNet(cfgs, self.image_dim, device)
        self.v2e2v_net.load_state_dict(checkpoint['state_dict'], strict=True)
        
        self.v2e2v_net.to(device)
        self.v2e2v_net.eval()
        
        
    def forward(self):
        with torch.no_grad():
            for seq_id, path_to_sequence_folder in enumerate(self.path_to_sequences):
                dataset_name=path_to_sequence_folder.split('/')[-1].split('.')[0]
                if self.test_data_name is not None and dataset_name != self.test_data_name:
                    continue
                self.video_renderer.initialize(path_to_sequence_folder, self.num_load_frames)
                num_packs = int(np.floor(self.video_renderer.num_frames / (self.num_pack_frames-1)))-1

                print('Number of frames in sequence {}: {} \n Number of frames per reconstruction: {} '.format(path_to_sequence_folder, self.video_renderer.num_frames, self.num_pack_frames))
                states = None
                prev_image = None
                num_events = 0
                
                image_writer = ImageWriter(cfgs, self.model_name, dataset_name)
                event_writer = EventWriter(cfgs, self.model_name, dataset_name)

                for frame_idx in range(num_packs):
                    frames, gt_frame, timestamps = self.video_renderer.update_frame_pack(self.num_pack_frames)
                    
                    if frames.shape[0] <= 1:
                        continue
                    
                    frames = torch.unsqueeze(torch.from_numpy(frames), axis=0).float().to(self.device)
                    timestamps = torch.unsqueeze(torch.from_numpy(timestamps), axis=0).to(self.device)
                    
                    pred_image, states = self.v2e2v_net(frames, timestamps, prev_image, states, seq_id)
                    prev_image = pred_image.detach()
                    
                    pred_image_numpy = pred_image.squeeze().cpu().data.numpy()
                    pred_image_numpy = np.uint8(cv2.normalize(pred_image_numpy, None, 0, 255, cv2.NORM_MINMAX))

                    image_writer(pred_image_numpy, frame_idx+1)
                    event_preview = make_event_preview(self.v2e2v_net.event_voxel_grids.cpu().numpy(), mode='red-blue', num_bins_to_show=-1) #'red-blue'
                    event_writer(event_preview, frame_idx+1)
                    
                    num_events += self.v2e2v_net.num_events
                
                print('Avg number of events per reconstruction: {:.1f}'.format(num_events/num_packs))
                
                    

if __name__ == '__main__':
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('device: ', device)
    
    parser = argparse.ArgumentParser(description='testing options')
    set_configs(parser)
    cfgs = parser.parse_args()
    
    v2e2v_testing = V2E2V(cfgs, device)
    v2e2v_testing()
    
    

