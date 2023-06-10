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
import cv2
import argparse


from utils.image_process import normalize_image
from data_readers.video_readers import VideoInterpolator, ImageReader
from e2v.e2v_model import CistaLSTCNet, CistaTCNet 
from utils.configs import set_configs
from utils.data_io import ImageWriter, EvalWriter
from utils.evaluate import mse, psnr, ssim, PerceptualLoss



class Reconstructor(nn.Module):
    def __init__(self, cfgs, device):
        super(Reconstructor, self).__init__()
        self.image_dim = cfgs.image_dim
        self.reader_type = cfgs.reader_type
        self.model_mode = cfgs.model_mode
        self.device = device
        self.num_load_frames = cfgs.test_img_num
        self.test_data_name = cfgs.test_data_name
        self.limit_num_events = cfgs.num_events
        self.test_data_mode = cfgs.test_data_mode 
        

        self.path_to_sequences = []
        for folder_name in os.listdir(cfgs.path_to_test_data):
            if os.path.isdir(os.path.join(cfgs.path_to_test_data, folder_name)):
                self.path_to_sequences.append(os.path.join(cfgs.path_to_test_data, folder_name))
        self.path_to_sequences.sort()
        if self.reader_type == 'upsampling':
            self.video_renderer = VideoInterpolator(self.image_dim, num_bins=cfgs.num_bins, is_with_events=True, time_unit=cfgs.time_unit, device=self.device)
        else:
            self.video_renderer = ImageReader(self.image_dim, num_bins=cfgs.num_bins, is_with_events=True, time_unit=cfgs.time_unit)
            
        # initialize reconstruction network        
        if self.model_mode == 'cista-lstc':
            self.model = self.e2v_net = CistaLSTCNet(image_dim=self.image_dim, base_channels=cfgs.base_channels, depth=cfgs.depth, num_bins=cfgs.num_bins)
        elif self.model_mode == 'cista-tc':
            self.model = CistaTCNet(image_dim=self.image_dim, base_channels=cfgs.base_channels, depth=cfgs.depth, num_bins=cfgs.num_bins)
        else:
            assert self.model_mode in ['cista-lstc', 'cista-tc'], "Model should be 'cista-lstc' and 'cista-tc'! "
        
        # Load pretrained model
        self.model_name = os.path.splitext(cfgs.path_to_test_model.split('/')[-1])[0]
        checkpoint = torch.load(cfgs.path_to_test_model, map_location='cuda:0')
        self.model.load_state_dict(checkpoint['state_dict'], strict=True)
        print(self.model)
        
        self.model.to(device)
        self.model.eval()


        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        self.lpips_fn = PerceptualLoss(net='vgg', device=device)  
    
    def evaluate(self, pred_image, gt_image):  
        test_mse = mse(pred_image/255. , gt_image)
        test_psnr = psnr(pred_image/255. , gt_image)
        test_ssim = ssim(pred_image/255. , gt_image)
        pred = torch.from_numpy(pred_image/255).float().to(self.device)
        gt = torch.from_numpy(gt_image).float().to(self.device)
        test_lpips = self.lpips_fn(pred, gt, normalize=True).item() 

        return [test_mse, test_psnr, test_ssim, test_lpips] 
       
       
        
    def forward(self):
        with torch.no_grad():
            for seq_id, path_to_sequence_folder in enumerate(self.path_to_sequences):
                dataset_name=path_to_sequence_folder.split('/')[-1].split('.')[0]

                if self.test_data_name is not None and dataset_name != self.test_data_name:
                    continue
                self.video_renderer.initialize(path_to_sequence_folder, self.num_load_frames)

                states = None
                prev_image = None

                image_writer = ImageWriter(cfgs, self.model_name, dataset_name)
                eval_writer = EvalWriter(cfgs, self.model_name, dataset_name)

                all_test_results = []
                
                frame_idx = 0
                while not self.video_renderer.ending:
                # for frame_idx in range(num_packs):
                    events, gt_frame = self.video_renderer.update_event_frame_pack(self.limit_num_events, self.test_data_mode)

                    if prev_image is None:
                        prev_image = torch.zeros([1, 1, self.image_dim[0], self.image_dim[1]], dtype=torch.float32, device=self.device)
                    
                    for evs in events:
                        evs = torch.unsqueeze(torch.from_numpy(evs), axis=0).to(self.device)
                        
                        pred_image, states = self.model(evs, prev_image, states)
                        prev_image = pred_image.detach()
                    
                    
                    pred_image = pred_image.squeeze()
                    pred_image_numpy = pred_image.cpu().data.numpy()
                   
                    pred_image_numpy = np.uint8(cv2.normalize(pred_image_numpy, None, 0, 255, cv2.NORM_MINMAX)) # HQF
                    # pred_image_numpy = np.uint8(normalize_image(pred_image.cpu().data)*255) # ECD
                    gt_image_norm = normalize_image(torch.from_numpy(gt_frame).float())
                    gt_image_norm = gt_image_norm.squeeze().cpu().data.numpy()


                    image_writer(pred_image_numpy, frame_idx+1)

                    all_test_results.append(self.evaluate(pred_image_numpy, gt_image_norm))
                    
                    frame_idx += 1

                all_test_results = np.array(all_test_results)
                mean_test_results = all_test_results.mean(0)
                
                mean_results = [eval_writer.dataset_name] + list(np.array(mean_test_results).round(4)) + [len(all_test_results)]

                print('\nTest set {}: Average MSE for {:d} frames: {:.4f}, PSNR: {:.4f}, SSIM: {:.4f}, LPIPS: {:.4f} \n'.format(
                    eval_writer.dataset_name, len(all_test_results), mean_test_results[0], mean_test_results[1], mean_test_results[2], mean_test_results[3]))
                
                name_results = ['Dataset', 'MSE', 'PSNR', 'SSIM', 'LPIPS', 'N_frames']
                eval_writer(name_results, mean_results)
            
                    

if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('device: ', device)
    
    parser = argparse.ArgumentParser()
    set_configs(parser)
    cfgs = parser.parse_args()
    
    reconstuctor = Reconstructor(cfgs, device)
    reconstuctor()
    
