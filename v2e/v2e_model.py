"""
These codes are modified from 
https://github.com/SensorsINI/v2e/blob/master/v2ecore/emulator.py

DVS simulator.
Compute events from input frames.
"""

import random
import math

import cv2
import numpy as np
import logging
import matplotlib.pyplot as plt

import torch
import torch.nn as nn

# from .emulator_utils import low_pass_filter
from .emulator_utils import rescale_intensity_frame, lin_log, \
    low_pass_filter, subtract_leak_current, generate_shot_noise
from utils.event_process import *


# import rosbag # not yet for python 3

logger = logging.getLogger(__name__)



class EventEmulator(nn.Module):
    """compute events based on the input frame.
    """

    def __init__(
            self,
            output_mode,
            pl=1, # large coef for C
            ps=1, # small coef for C
            ql=1, # large coef for fc
            qs=1, # small coef for fc
            num_bins = 5,
            pos_thres=0.2,
            neg_thres=0.2,
            sigma_thres=0.03,
            cutoff_hz = 0,
            leak_rate_hz=0.1,
            refractory_period_s=0,
            shot_noise_rate_hz=0,  # rate in hz of temporal noise events
            leak_jitter_fraction=0.1, #0.1,
            noise_rate_cov_decades=0.1, #0.1,
            seed=0,
            # change as you like to see 'baseLogFrame',
            # 'lpLogFrame', 'diff_frame'
            show_dvs_model_state: str = None,
            device="cuda"):
        super(EventEmulator, self).__init__()
        """
        Parameters
        ----------
        output_mode: str, 'raw' or 'voxel_grid'
            Output raw events (torch.tensor) [N, 5] (timestamp,x,y,polarity,batch_size) 
            or event voxel grid (torch.tensor) [batch_size, num_bins, H, W]
        pl: float, default 1
            Coefficeint of contrast threshold for a large portion of pixels, Cl = pl*C
        ps: float, default 1
            Coefficeint of contrast threshold for a small portion of pixels, Cs = ps*C
        pl: float, default 1
            Coefficeint of cutoff frequency for a large portion of pixels, fc_l = ql*C
        ps: float, default 1
            Coefficeint of cutoff frequency for a small portion of pixels, fc_s = qs*C
        num_bins: int, default 5
            number of bins for event voxel grid
        pos_thres: float, default 0.2
            nominal threshold of triggering positive event in log intensity.
        neg_thres: float, default 0.2
            nominal threshold of triggering negative event in log intensity.
        sigma_thres: float, default 0.03
            std deviation of threshold in log intensity.
        cutoff_hz: float, default 0
            3dB cutoff frequency in Hz of DVS photoreceptor, no temporal filtering if <=0
        leak_rate_hz: float, default 0.1
            leak event rate per pixel in Hz,
            from junction leakage in reset switch
        refractory_period_s: float, default 0
            refractory period in seconds
        shot_noise_rate_hz: float, default 0
            shot noise rate in Hz
        leak_jitter_fraction: float, default 0.1
        noise_rate_cov_decades: float, default 0.1
        seed: int, default=0
            seed for random threshold variations,
            fix it to nonzero value to get same mismatch every time
        show_dvs_model_state: str,
            None or 'new_frame' 'baseLogFrame','lpLogFrame0','lpLogFrame1',
            'diff_frame'
        
        """

        logger.info(
            "ON/OFF log_e temporal contrast thresholds: "
            "{} / {} +/- {}".format(pos_thres, neg_thres, sigma_thres))
        # self.repeat = repeat
        self.base_log_frame = None
        self.t_previous = None  # time of previous frame
        self.lp_log_frame0 = None

        # torch device
        self.device = device

        self.output_mode = output_mode # raw or voxel_grid
        self.num_bins = num_bins # for voxel grid
        
        # cutoff frequency
        self.ql = ql
        self.qs = qs
        self.cutoff_hz = cutoff_hz
        
        # thresholds
        self.pl = pl
        self.ps = ps
        self.sigma_thres = sigma_thres
        # initialized to scalar, later overwritten by random value array
        self.pos_thres = torch.tensor(pos_thres, dtype=torch.float32, device=self.device)
        # initialized to scalar, later overwritten by random value array
        self.neg_thres = torch.tensor(neg_thres, dtype=torch.float32, device=self.device)
        self.pos_thres_nominal = torch.tensor(pos_thres, dtype=torch.float32, device=self.device)
        self.neg_thres_nominal = torch.tensor(neg_thres, dtype=torch.float32, device=self.device)
        
        # non-idealities
        self.leak_rate_hz = leak_rate_hz
        self.refractory_period_s = torch.tensor(refractory_period_s, dtype=torch.float32, device=self.device)
        self.shot_noise_rate_hz = shot_noise_rate_hz

        self.leak_jitter_fraction = leak_jitter_fraction
        self.noise_rate_cov_decades = noise_rate_cov_decades

        self.SHOT_NOISE_INTEN_FACTOR = 0.25

        # output properties
        self.show_input = show_dvs_model_state

        # generate jax key for random process
        if seed != 0:
            torch.manual_seed(seed)
            np.random.seed(seed)
            random.seed(seed)


        # event stats
        self.num_events = 0
        self.frame_counter = 0


    def _init(self, frame_log, Tr_frames):
        '''Initialise base frame and other parameters
        frame_linear: torch.tensor
        Tr_frames: 
            refractory period in the format as frames
        
        '''
        logger.debug(
            'initializing random temporal contrast thresholds '
            'from from base frame')
        
        # base_frame are memorized lin_log pixel values
        # The first
        
        self.base_log_frame = frame_log if self.base_log_frame is None else  self.base_log_frame #lin_log(first_frame_linear)
        
        # # initialize first stage of 2nd order IIR to first input
        
        self.lp_log_frame0 = self.base_log_frame.clone().detach() if self.lp_log_frame0 is None else self.base_log_frame

        # Initialise pos/neg threshilds for each pixel
        # take the variance of threshold into account.
        # print('Pos/Neg contrast threshold: {}, {}\n'.format(self.pos_thres_real_nominal.cpu().numpy(), self.neg_thres_real_nominal.cpu().numpy()))
        self.pos_thres_real_nominal = self.pos_thres_nominal*torch.ones(frame_log.size()[0], dtype=torch.float32, device=self.device)
        self.neg_thres_real_nominal = self.neg_thres_nominal*torch.ones(frame_log.size()[0], dtype=torch.float32, device=self.device)
        
        if self.sigma_thres > 0:
            self.pos_thres = [torch.normal(
                self.pl*self.pos_thres_real_nominal[i], self.sigma_thres,
                size=list(frame_log[i,0:1].shape),
                dtype=torch.float32).to(self.device) for i in range(len(self.pos_thres_real_nominal))] #*self.num_repeat 
            self.pos_thres = torch.stack(self.pos_thres, 0) # N x N_repeat x 1 x HxW
            
            self.pos_thres_half = [torch.normal(
                self.ps*self.pos_thres_real_nominal[i], self.sigma_thres,
                size=list(frame_log[i,0:1].shape),
                dtype=torch.float32).to(self.device) for i in range(len(self.pos_thres_real_nominal))] #*self.num_repeat 
            self.pos_thres_half = torch.stack(self.pos_thres_half, 0)
            self.pos_thres[:,:,0::2, 0::2] = self.pos_thres_half[:,:,0::2, 0::2]
            # self.pos_thres[:,:,1::2, 1::2] = self.pos_thres_half[:,:,1::2, 1::2]

            # to avoid the situation where the threshold is too small.
            self.pos_thres = torch.clamp(self.pos_thres, min=0.01)

            self.neg_thres = [torch.normal(
                self.pl*self.neg_thres_real_nominal[i], self.sigma_thres,
                size=list(frame_log[i,0:1].shape),
                dtype=torch.float32).to(self.device) for i in range(len(self.neg_thres_real_nominal))] #*self.num_repeat 
            self.neg_thres = torch.stack(self.neg_thres, 0)
            
            self.neg_thres_half = [torch.normal(
                self.ps*self.neg_thres_real_nominal[i], self.sigma_thres,
                size=list(frame_log[i,0:1].shape),
                dtype=torch.float32).to(self.device) for i in range(len(self.neg_thres_real_nominal))] #*self.num_repeat 
            self.neg_thres_half = torch.stack(self.neg_thres_half, 0)
            self.neg_thres[:,:,0::2, 0::2] = self.neg_thres_half[:,:,0::2, 0::2]
            # self.neg_thres[:,:,1::2, 1::2] = self.neg_thres_half[:,:,1::2, 1::2]
            
            self.neg_thres = torch.clamp(self.neg_thres, min=0.01)

        # compute variable for shot-noise
        self.pos_thres_pre_prob = torch.einsum('i,ijkl->ijkl', 1./self.pos_thres_real_nominal, self.pos_thres)
        self.neg_thres_pre_prob = torch.einsum('i,ijkl->ijkl', 1./self.neg_thres_real_nominal, self.neg_thres)

        # If leak is non-zero, then initialize each pixel memorized value
        # some fraction of ON threshold below first frame value, to create leak
        # events from the start; otherwise leak would only gradually
        # grow over time as pixels spike.
        # do this *AFTER* we determine randomly distributed thresholds
        # (and use the actual pixel thresholds)
        # otherwise low threshold pixels will generate
        # a burst of events at the first frame
        if self.leak_rate_hz > 0:
            # no justification for this subtraction after having the
            # new leak rate model
            # set noise rate array, it's a log-normal distribution
            self.noise_rate_array = torch.randn(
                frame_log[:,0:1].shape, dtype=torch.float32,
                device=self.device)
            self.noise_rate_array = torch.exp(
                math.log(10)*self.noise_rate_cov_decades*self.noise_rate_array)

        # refractory period
        if (self.refractory_period_s > 0).any():
            self.timestamp_mem = torch.add(torch.zeros_like(self.base_log_frame), -Tr_frames)
    
    def reset(self):
        '''resets so that next use will reinitialize the base frame
        '''
        self.num_events_total = 0
        self.num_events_on = 0
        self.num_events_off = 0
        self.base_log_frame = None
        self.lp_log_frame0 = None  # lowpass stage 0
        self.frame_counter = 0

    def IIR_temporal_filtering(self, log_frames, rescaled_frames, timestamps):
        # Apply nonlinear lowpass filter here.
        # Filter is a 1st order lowpass IIR (can be 2nd order)
        if self.cutoff_hz > 0:
            num_frames = log_frames.size()[1]

            LP_features= [self.lp_log_frame0]
            delta_time = timestamps[1:] - timestamps[:-1]
            # lp_log_frame0 = log_frames[:,0:1].clone().detach()
           
            # ignore the first frame, since its the last frame in the last generation
            for n in range(1, num_frames): # 1,num_frames
                self.lp_log_frame0 = low_pass_filter(
                                log_new_frame=log_frames[:,n],
                                lp_log_frame0=self.lp_log_frame0,
                                inten01=rescaled_frames[:,n:n+1],
                                delta_time=delta_time[n-1], #n-1
                                cutoff_hz=self.cutoff_hz,
                                qs=self.qs,
                                ql=self.ql)
                LP_features.append(self.lp_log_frame0)
            return torch.cat(LP_features, 1)
        else:
            return log_frames
  
  
    def _show(self, inp: np.ndarray):
        inp = np.array(inp.cpu().data[0,0,:,:].numpy())
        min = np.min(inp)
        norm = (np.max(inp) - min)
        if norm == 0:
            logger.warning('image is blank, max-min=0')
            norm = 1
        img = ((inp - min) / norm)
        cv2.imshow(self.show_input, img) #__name__+':'+
        cv2.waitKey(30)


    def forward(self, frames, t_frames): #new_frames, rescaled_org_frames, t_frames
        """Compute events based on a sequence of frames.
        Inputs:
            frames: torch.tensor, [batch_size=1, num_frames, height, width]
            Input frames (at least 2 frames) must be high frame rate or upsampled frames
            t_frames: list of float, [batch_size=1, 2] or [batch_size, num_frames]
                timestamp of the first and the end frame in float seconds or timestamps of all the frames
        Outputs:
            events: torch.tensor [N, 5] for raw events or [batch_size=1, num_bins, height, width] for voxel grid
            [N, 5], each row contains [timestamp, x, y, polarity, batch_size].

        """
        # update frame counter
        batch_size, num_frames, height, width = frames.size()
        self.frame_counter += num_frames

        # time frames
        if t_frames.size()[1] == 2:
            t_float_frames = torch.linspace(start=t_frames[0,0], end=t_frames[0,-1], \
                steps=num_frames, dtype=torch.float32, device=self.device)
        else:
            t_float_frames = t_frames[0]
        # Time for event voxel grid
        duration = (self.num_bins - 1)/(num_frames-1)
        time_frames = torch.linspace(start=0, end=duration*(num_frames-1), steps=num_frames, dtype=torch.float32, device=self.device)
        
        # num_repeat = int(batch_size / len(self.refractory_period_s)) 
        Tr = torch.einsum('i,ij->ij', (self.num_bins-1)*self.refractory_period_s.repeat(batch_size), 1/(t_frames[:,-1:]- t_frames[:,0:1])).float()
        Tr_frames = Tr.repeat(height, width,1,1).permute(2,3,0,1)
        
        
        frames_rescaled = rescale_intensity_frame(frames) #(I+20)/275
        frames_log = lin_log(frames)
        
        # transform refractory period for voxel grid 
        if self.base_log_frame is None:
            self._init(frames_log[:,0:1,:,:], Tr_frames) #, frames_rescaled)
            self.t_previous = t_frames[0,0]
        else:
            self.timestamp_mem[self.timestamp_mem>0] -= (self.num_bins-1)
            self.timestamp_mem[self.timestamp_mem<0] = -Tr_frames[self.timestamp_mem<0]

        frames_filtered = self.IIR_temporal_filtering(frames_log, frames_rescaled, t_float_frames)
        

        if t_float_frames[1] <= self.t_previous:
            raise ValueError(
                "this frame time={} must be later than "
                "previous frame time={}".format(t_float_frames[1], self.t_previous))

        # lin-log mapping
        

        num_events = 0
        if self.output_mode == 'voxel_grid':
            events = torch.zeros((batch_size, self.num_bins, height, width), device=self.device).flatten()
        else:
            events = torch.tensor([], device=self.device)
        

        for n in range(1, num_frames):
            new_frame = frames_filtered[:,n:n+1,...]
            # compute time difference between this and the previous frame
            delta_time = t_float_frames[n] - self.t_previous
            
            # Leak events: switch in diff change amp leaks at some rate
            # equivalent to some hz of ON events.
            # Actual leak rate depends on threshold for each pixel.
            # We want nominal rate leak_rate_Hz, so
            # R_l=(dI/dt)/Theta_on, so
            # R_l*Theta_on=dI/dt, so
            # dI=R_l*Theta_on*dt
            if self.leak_rate_hz > 0:
                self.base_log_frame = subtract_leak_current(
                    base_log_frame=self.base_log_frame,
                    leak_rate_hz=self.leak_rate_hz,
                    delta_time=delta_time,
                    pos_thres=self.pos_thres,
                    leak_jitter_fraction=self.leak_jitter_fraction,
                    noise_rate_array=self.noise_rate_array)

            # log intensity (brightness) change from memorized values is computed
            # from the difference between new input
            # (from lowpass of lin-log input) and the memorized value
            # self.lp_log_frame1 - self.base_log_frame ###########
            diff_frame = new_frame - self.base_log_frame


            if self.show_input:
                if self.show_input == 'new_frame':
                    self._show(new_frame)
                elif self.show_input == 'baseLogFrame':
                    self._show(self.base_log_frame)
                elif self.show_input == 'lpLogFrame0':
                    self._show(self.lp_log_frame0)
                elif self.show_input == 'lpLogFrame1':
                    self._show(self.lp_log_frame1)
                elif self.show_input == 'diff_frame':
                    self._show(diff_frame)
                else:
                    logger.error("don't know about showing {}".format(
                        self.show_input))

            # generate event map
            mask_tolerence = diff_frame.abs() > 1e-6
            diff_frame.masked_fill_(~mask_tolerence, 0.)

            pol = torch.zeros_like(diff_frame)
            pol.masked_fill_(diff_frame > 0, 1.0)
            pol.masked_fill_(diff_frame < 0, -1.0)

            # Compute event count for each pixel, the max count is the num_iters
            # Update max one event per pixel at each iteration
            C = torch.mul(self.pos_thres, pol>0) + torch.mul(self.neg_thres,pol<0) 
            event_counts = torch.div(diff_frame.abs(), C+1e-9).floor().type(torch.int32)
            num_iters= event_counts.max(2)[0].max(2)[0].squeeze(1) #(batch_size, 1)
            max_num_iters = num_iters.max()

            # event timestamps at each iteration
            # intermediate timestamps are linearly spaced
            # they start after the t_start to make sure
            # that there is space from previous frame
            # they end at t_end
            # e.g. t_start=0, t_end=1, num_iters=2, i=0,1
            # ts=1*1/2, 2*1/2
            #  ts = self.t_previous + delta_time * (i + 1) / num_iters
            num_iters[num_iters==0] = 1
            ts_step =  duration / num_iters
            steps = torch.linspace(start=1, end=max_num_iters, steps=max_num_iters, dtype=torch.float32, device=self.device).repeat(batch_size, 1)
            ts = time_frames[n-1] + torch.einsum('i,ij->ij', ts_step, steps)
            for m in range(batch_size):
                ts[m, num_iters[m]:] = 0
            ts_frames = ts.repeat(height, width,1,1).permute(2,3,0,1)
            final_evts_frame = torch.zeros(pol.shape, dtype=torch.int32, device=self.device)
            t = torch.zeros(pol.shape, dtype=torch.float32, device=self.device)
            
            # NOISE: add temporal noise here by
            # simple Poisson process that has a base noise rate
            # self.shot_noise_rate_hz.
            # If there is such noise event,
            # then we output event from each such pixel

            # the shot noise rate varies with intensity:
            # for lowest intensity the rate rises to parameter.
            # the noise is reduced by factor
            # SHOT_NOISE_INTEN_FACTOR for brightest intensities
            # This was in the loop, here we calculate loop-independent quantities
            # self.shot_noise_rate_hz=-1
            if self.shot_noise_rate_hz > 0:
                shot_on_cord, shot_off_cord = generate_shot_noise(
                        shot_noise_rate_hz=self.shot_noise_rate_hz,
                        delta_time=delta_time,
                        num_iters=num_iters,
                        shot_noise_inten_factor=self.SHOT_NOISE_INTEN_FACTOR,
                        inten01=frames_rescaled[:,n:n+1],
                        pos_thres_pre_prob=self.pos_thres_pre_prob,
                        neg_thres_pre_prob=self.neg_thres_pre_prob)
                shot_on_off_cord = torch.mul(shot_on_cord, pol>0) + torch.mul(shot_off_cord, pol<0)

            for i in range(max_num_iters):
                # already have the number of events for each pixel in
                # pos_evts_frame, just find bool array of pixels with events in
                # this iteration of max # events

                # it must be >= because we need to make event for
                # each iteration up to total # events for that pixel
                mask_gen_events = (event_counts >= i+1)

                # generate shot noise
                if self.shot_noise_rate_hz > 0:
                    # update event list
                    mask_gen_events = torch.logical_or(mask_gen_events, shot_on_off_cord[i])

                # update the base log frame, along with the shot noise
                # final_evts_frame += mask_gen_events
                
                # filter events with refractory_period
                # only filter when refractory_period_s is large enough
                # otherwise, pass everything
                if (Tr > ts_step).any():
                    time_since_last_spike = ts_frames[:,i:i+1,...]*mask_gen_events-self.timestamp_mem
                    mask_gen_events = time_since_last_spike > Tr_frames #.squeeze()  
                    self.timestamp_mem[mask_gen_events] =  ts_frames[:,i:i+1,...][mask_gen_events]
                
                # # update the base log frame, along with the shot noise
                final_evts_frame += mask_gen_events
                
                if self.output_mode == 'voxel_grid':                    
                    t = ts_frames[:,i:i+1,...]*mask_gen_events 
                    ti = t.floor()
                    ti_long = t.long()
                    dts = t - ti
                    vals_left = torch.mul(pol, (1.0 - dts))
                    vals_right = torch.mul(pol, dts)

                    t_mask = mask_gen_events & (ti >=0)
                    batch_indices, _, y_indices, x_indices = t_mask.nonzero(as_tuple=True)
                    num_events += len(y_indices)

                    events.index_add_(dim=0,
                        index=x_indices + y_indices * width + 
                        ti_long[t_mask] * width * height + 
                        batch_indices * self.num_bins * width * height,
                        source=vals_left[t_mask])
                
                    t_mask &= ((ti + 1) < self.num_bins)
                    
                    batch_indices, _, y_indices, x_indices = t_mask.nonzero(as_tuple=True)
                    events.index_add_(dim=0,
                        index=x_indices + y_indices * width + 
                        (ti_long[t_mask]+1) * width * height + 
                        batch_indices * self.num_bins * width * height,
                        source=vals_right[t_mask])
                                
                else: # self.output_mode == 'raw': 
                    t = torch.mul(ts_frames[:,i:i+1,...], mask_gen_events)
                
                    # generate events
                    # make a list of coordinates x,y addresses of events
                    # b,1,y,x
                    event_xy = mask_gen_events.nonzero(as_tuple=True)
                    num_events += len(event_xy[0])

                    t_1d = t[event_xy] 
                    pol_1d = pol[event_xy]
                        
                    #[t,x,y,p,b]
                    cur_events = torch.stack((t_1d, event_xy[3], event_xy[2], pol_1d, event_xy[0]), 1)
                    events = torch.cat((events, cur_events), 0)
                    
  
            self.t_previous = t_float_frames[n]
            self.base_log_frame += pol*final_evts_frame*C

        if self.output_mode == 'voxel_grid':
            events = events.view(batch_size, self.num_bins, height, width)
            events = event_preprocess_pytorch(events, mode='std', filter_hot_pixel=False)
        else:
            if len(events) != 0:
               # along timestamp
               _, indices = torch.sort(events[:,0], dim=0, descending=False, out=None)
               events = events[indices]
               # along batch
               _, indices = torch.sort(events[:,-1], dim=0, descending=False, out=None)
               events = events[indices]

        return events, num_events




