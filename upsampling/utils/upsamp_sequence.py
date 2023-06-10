''' Adaptive Upsampling: modified from
https://github.com/uzh-rpg/rpg_vid2e/tree/master/upsampling
Upsampler: online upsample video sequence 
for loaded frames and timestamps of an sequence <img_sequence, time_sequence>
'''


import os
from typing import List
import urllib
import warnings

import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image

from .const import mean, std
from .model import UNet, backWarp
from .dataset import CropParameters


class Upsampler:
    def __init__(self, image_dim, is_train, device: str):
        self.crop = CropParameters(image_dim[1], image_dim[0], 5)
        self.is_not_train = not is_train
        # if self.is_not_train:
        self.device = torch.device(device)
        

        self._load_net_from_checkpoint()

        negmean= [x * -1 for x in mean]
        self.negmean = torch.Tensor([x * -1 for x in mean]).view(3, 1, 1)
        if self.is_not_train:
            self.negmean = self._move_to_device(self.negmean, self.device)
        revNormalize = transforms.Normalize(mean=negmean, std=std)
        self.TP = transforms.Compose([revNormalize])

        normalize = transforms.Normalize(mean=mean, std=std)
        self.transform = transforms.Compose([transforms.ToTensor(), normalize])

    def _load_net_from_checkpoint(self):
        ckpt_file = 'upsampling/checkpoint/SuperSloMo.ckpt'

        if not os.path.isfile(ckpt_file):
            print('Downloading SuperSlowMo checkpoint to {} ...'.format(ckpt_file))
            # g = urllib.request.urlopen('http://rpg.ifi.uzh.ch/data/VID2E/SuperSloMo.ckpt')
            # with open(ckpt_file, 'w+b') as ckpt:
            #     ckpt.write(g.read())
            # print('Done with downloading!')
            import gdown
            id = '1YL2EnX0MsrH_5_PjhDr__c6NaT_y8I7Z'
            gdown.download(id=id, output=ckpt_file, quiet=False)
            
        assert os.path.isfile(ckpt_file)

        self.flowComp = UNet(6, 4)
        if self.is_not_train:
            self._move_to_device(self.flowComp, self.device)
        for param in self.flowComp.parameters():
            param.requires_grad = False
        self.ArbTimeFlowIntrp = UNet(20, 5)
        if self.is_not_train:
            self._move_to_device(self.ArbTimeFlowIntrp, self.device)
        for param in self.ArbTimeFlowIntrp.parameters():
            param.requires_grad = False

        self.flowBackWarp_dict = dict()

        checkpoint = torch.load(ckpt_file, map_location=self.device)
        self.ArbTimeFlowIntrp.load_state_dict(checkpoint['state_dictAT'])
        self.flowComp.load_state_dict(checkpoint['state_dictFC'])

    def get_flowBackWarp_module(self, width: int, height: int):
        module = self.flowBackWarp_dict.get((width, height))
        if module is None:
            module  = backWarp(width, height, self.device, (not self.is_not_train))
            if self.is_not_train:
                self._move_to_device(module, self.device)
            self.flowBackWarp_dict[(width, height)] = module
        assert module is not None
        return module


    def upsampling(self, img_sequence, time_sequence):
        # img_sequence: list of uint8, time_sequence: list
        timestamps_list = list()
        img_sequence = [self.transform(Image.fromarray((img).astype('uint8')).convert("RGB")) for img in img_sequence]
        if self.is_not_train:
            img_sequence = self._move_to_device(img_sequence, self.device)
        
        final_frames = []
        for i in range(len(img_sequence)-1):
            img_pair = img_sequence[i:i+2]
            time_pair = time_sequence[i:i+2]
            I0 = torch.unsqueeze(img_pair[0], dim=0)
            I1 = torch.unsqueeze(img_pair[1], dim=0)
            t0 = time_pair[0]
            t1 = time_pair[1]

            I0 = self.crop.pad(I0)
            I1 = self.crop.pad(I1)
            total_frames = []
            timestamps = []

            flowOut = self.flowComp(torch.cat((I0, I1), dim=1))##############
            F_0_1 = flowOut[:, :2, :, :]
            F_1_0 = flowOut[:, 2:, :, :]

            total_frames.append(self.TP(I0[0]))
            timestamps.append(t0)

            self._upsample_adaptive(I0, I1, t0, t1, F_0_1, F_1_0, total_frames, timestamps)

            total_frames.append(self.TP(I1[0]))
            timestamps.append(t1)
            sorted_indices = np.argsort(timestamps)

            total_frames = torch.stack([total_frames[j] for j in sorted_indices])
            timestamps = [timestamps[i] for i in sorted_indices]
            total_frames = self._to_numpy_image(total_frames)

            if i != len(img_sequence)-2:
                total_frames = total_frames[:-1]
                timestamps = timestamps[:-1]
            for _, frame in enumerate(total_frames):
                frame = frame[self.crop.iy0:self.crop.iy1,
                                            self.crop.ix0:self.crop.ix1, :]
                final_frames.append(np.uint8(255.*cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)))
            timestamps_list += timestamps
        return np.array(final_frames), np.array(timestamps_list)

        
    @staticmethod
    def _to_numpy_image(img: torch.Tensor):
        img = np.clip(img.cpu().numpy(), 0, 1) #.astype(np.uint8) # *255
        img = np.transpose(img, (0, 2, 3, 1))
        return img

    def _upsample_adaptive(self,
                           I0: torch.Tensor,
                           I1: torch.Tensor,
                           time0: torch.Tensor,
                           time1: torch.Tensor,
                           F_0_1: torch.Tensor,
                           F_1_0: torch.Tensor,
                           total_frames: List[torch.Tensor],
                           timestamps: List[float]):
        B, _, _, _ = F_0_1.shape

        flow_mag_0_1_max, _ = F_0_1.pow(2).sum(1).pow(.5).view(B,-1).max(-1)
        flow_mag_1_0_max, _ = F_1_0.pow(2).sum(1).pow(.5).view(B,-1).max(-1)

        flow_mag_max, _ = torch.stack([flow_mag_0_1_max, flow_mag_1_0_max]).max(0)
        flow_mag_max = torch.ceil(flow_mag_max).int()

        for i in range(B):
            for intermediateIndex in range(1, flow_mag_max[i].item()): #flow_mag_max[i].item() self.num_interp_frames
                t = float(intermediateIndex) / flow_mag_max[i].item()#flow_mag_max[i].item()
                temp = -t * (1 - t)
                fCoeff = [temp, t * t, (1 - t) * (1 - t), temp]

                F_t_0 = fCoeff[0] * F_0_1 + fCoeff[1] * F_1_0
                F_t_1 = fCoeff[2] * F_0_1 + fCoeff[3] * F_1_0

                height, width = I0.shape[-2:]
                flow_back_warp = self.get_flowBackWarp_module(width, height)
                g_I0_F_t_0 = flow_back_warp(I0, F_t_0)
                g_I1_F_t_1 = flow_back_warp(I1, F_t_1)

                intrpOut = self.ArbTimeFlowIntrp(
                    torch.cat((I0, I1, F_0_1, F_1_0, F_t_1, F_t_0, g_I1_F_t_1, g_I0_F_t_0), dim=1))
                F_t_0_f = intrpOut[:, :2, :, :] + F_t_0
                F_t_1_f = intrpOut[:, 2:4, :, :] + F_t_1
                V_t_0 = torch.sigmoid(intrpOut[:, 4:5, :, :])
                V_t_1 = 1 - V_t_0

                g_I0_F_t_0_f = flow_back_warp(I0, F_t_0_f)
                g_I1_F_t_1_f = flow_back_warp(I1, F_t_1_f)

                wCoeff = [1 - t, t]

                Ft_p = (wCoeff[0] * V_t_0 * g_I0_F_t_0_f + wCoeff[1] * V_t_1 * g_I1_F_t_1_f) / (
                        wCoeff[0] * V_t_0 + wCoeff[1] * V_t_1)

                Ft_p_norm = Ft_p[i] - self.negmean

                total_frames += [Ft_p_norm]
                timestamps += [(time0 + t * (time1 - time0))]

    @classmethod
    def _move_to_device(
            cls,
            _input,
            device: torch.device,
            dtype: torch.dtype = None):
        if not torch.cuda.is_available() and not device == torch.device('cpu'):
            warnings.warn("CUDA not available! Input remains on CPU!", Warning)

        if isinstance(_input, torch.nn.Module):
            # Performs in-place modification of the module but we still return for convenience.
            return _input.to(device=device, dtype=dtype)
        if isinstance(_input, torch.Tensor):
            return _input.to(device=device, dtype=dtype)
        if isinstance(_input, list):
            return [cls._move_to_device(v, device=device, dtype=dtype) for v in _input]
        warnings.warn("Instance type '{}' not supported! Input remains on current device!".format(type(_input)), Warning)
        return _input


