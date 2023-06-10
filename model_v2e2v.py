import torch.nn as nn
import matplotlib.pyplot as plt

from v2e.v2e_model import EventEmulator
from e2v.e2v_model import CistaLSTCNet
from utils.event_process import *


class V2E2VNet(nn.Module):
    def __init__(self, cfgs, image_dim, device): #e2v_net
        super(V2E2VNet, self).__init__()
        ''' Video-to-events-to-video framework, 
        including V2E generation and E2V (CISTA-LSTC) reconstruction network
        Input frames must be high frame rate / upsampled
        A set of frames --> events / event voxel grid --> reconstructed frame (Ground Truth: the last frame of the input)
        Events (Event voxel grids) are generated from a set of frames for each reconstruction
        ------ Parameters ------
        cfgs: configs
        image_dim: tuple/list, [height, width]
        device: str, 'cuda:0' or 'cpu'
        '''
        self.height, self.width = image_dim
        self.device = device
        self.display = cfgs.display_train
        
        # event_mode has to be 'voxel grid'
        self.event_mode = cfgs.event_mode
        # number of bins of event voxel grid
        self.num_bins = cfgs.num_bins 
        
        # initialise
        # ID of a sequence, v2e_net keeps updating for the same sequence
        # but reset when the sequence has changed 
        self.seq_id = -1
        self.img_id = 0
        # For monitering / saving data (num_events, event_voxel_grids)
        self.num_events = -1
        self.event_voxel_grids = None
        
        print('Coefficients for CT (C = {}) : ({}, {})'.format(cfgs.C, cfgs.pl, cfgs.ps))
        print('Coefficients for cutoff freq (fc = {}): ({}, {})'.format(cfgs.cutoff_hz, cfgs.ql, cfgs.qs))

            
        self.v2e_net = EventEmulator(
                                output_mode=self.event_mode,
                                num_bins=cfgs.num_bins,
                                pl = cfgs.pl,
                                ps = cfgs.ps,
                                ql = cfgs.ql,
                                qs = cfgs.qs,
                                pos_thres=cfgs.C,
                                neg_thres=cfgs.C,
                                sigma_thres=cfgs.threshold_sigma,
                                cutoff_hz = cfgs.cutoff_hz,
                                refractory_period_s = cfgs.refractory_period_s,
                                leak_rate_hz=0.1,
                                shot_noise_rate_hz=1,
                                device=device
                            )

        self.e2v_net = CistaLSTCNet(image_dim=image_dim, base_channels=cfgs.base_channels, depth=cfgs.depth, num_bins=cfgs.num_bins) 
  
    
    def reset_v2e(self, seq_idx):
        '''Reset v2e model if the sequence ID has changed'''
        if seq_idx != self.seq_id:
            self.v2e_net.reset()
            self.seq_id = seq_idx
            self.img_id = 0


    def forward(self, inputs, timestamps, pred_img, prev_states, seq_idx):
        '''
        Inputs:
            inputs: torch.tensor, float32, [batch_size=1, num_frames, H, W]
                Ground truth frames
            timestamps: float32 [batch_size=1, num_frames] or [batch_size=1, 2]
                List of timestamps for each frame, if only containing the start and the end timestamps, 
                timestamps for other frames are sampled linearly  
            pred_img: torch.tensor, float32, [batch_size=1, 1, H, W]
                Reconstructed frame in the last reconstruction, used in e2v_net
            prev_states: None or list of torch.tensor, float32
                Recurrent states in e2v_net
            seq_idx: int
                ID of a sequence, v2e_net keeps updating for the same sequence
                but reset when the sequence has changed 
        Outputs:
            rec_I: torch.tensor, float32, [batch_size=1, 1, H, W]
                Reconstructed frame, corresponding ground truth: the last frame of the inputs
            states: list of torch.tensor, float32
                Updated states in e2v_net
        '''
        
        if pred_img is None:
            pred_img = torch.zeros_like(inputs[:,0:1,:,:]).float()
        
        self.reset_v2e(seq_idx)
        self.img_id += 1
        
        # Event generation
        event_voxel_grids, cur_num_events = self.v2e_net(inputs, timestamps) #rescaled_inputs timestamps.repeat(frames_filtered.shape[0],1)
        
        # For evaluation
        self.num_events = cur_num_events
        self.event_voxel_grids = event_voxel_grids.clone().detach()
        
        # Image Reconstruction
        rec_I, states = self.e2v_net(event_voxel_grids, pred_img, prev_states) 
        
        if self.display:
            for i in range(4):
                plt.subplot(2,2,i+1)
                plt.axis('off')
                plt.imshow(event_voxel_grids.cpu().data[0,i,:,:], cmap='gray')
                plt.title('E%d' %i)
            plt.show()

            plt.subplot(1,2,1)
            plt.imshow(inputs.cpu().data[0,-1,...], cmap='gray')
            plt.axis('off')
            plt.title('GT')
            plt.subplot(1,2,2)
            plt.imshow(rec_I.cpu().data[0,0,...], cmap='gray')
            plt.axis('off')
            plt.title('I_rec')
            plt.show()
                
        return rec_I, states



