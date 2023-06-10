
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
import torch.utils.data as data
from torch import optim, nn 
from tensorboardX import SummaryWriter
import numpy as np
import argparse

from utils.configs import set_configs
from data_readers.train_data_loaders import TrainFixNEventData
from e2v.e2v_model import CistaLSTCNet, CistaTCNet
from utils.evaluate import PerceptualLoss
from pytorch_msssim import SSIM


class Train:
    def __init__(self, cfgs, device):
        # self.image_dim = cfgs.image_dim
        self.device = device
        
        self.model_name =  '{}_{}_b{}_d{}_c{}'.format(cfgs.model_name, cfgs.model_mode, \
                cfgs.num_bins, cfgs.depth, cfgs.base_channels)
        self.path_to_model = os.path.join(cfgs.path_to_model, self.model_name)
        if not os.path.exists(self.path_to_model):
            os.makedirs(self.path_to_model)
    

        if cfgs.model_mode == 'cista-lstc':
            self.model = self.e2v_net = CistaLSTCNet(image_dim=cfgs.image_dim, base_channels=cfgs.base_channels, depth=cfgs.depth, num_bins=cfgs.num_bins)
        elif cfgs.model_mode == 'cista-tc':
            self.model = CistaTCNet(image_dim=cfgs.image_dim, base_channels=cfgs.base_channels, depth=cfgs.depth, num_bins=cfgs.num_bins)
        else:
            assert self.model_mode in ['cista-lstc', 'cista-tc'], "Model_mode should be 'cista-lstc' and 'cista-tc'! "

        self.model = self.model.to(device)
        print(self.model)

        if cfgs.load_epoch_for_train:
            checkpoint = torch.load(os.path.join(self.path_to_model, '{}_{}.pth.tar'\
                                .format(self.model_name, cfgs.load_epoch_for_train)), map_location=self.device)
            self.model.load_state_dict(checkpoint['state_dict'], strict=True)

        
        # Load training data
        path_to_train_data = cfgs.path_to_train_data
        train_data = TrainFixNEventData(os.path.join(path_to_train_data, 'train_e2v.txt'), cfgs)
        self.train_loader = data.DataLoader(train_data,batch_size=cfgs.batch_size, shuffle=cfgs.shuffle, num_workers=4)       
        
        lr = cfgs.lr*(0.9**np.floor(cfgs.load_epoch_for_train/10.)) 
        self.optimizer = optim.Adam(self.model.parameters(),lr=lr)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=10, gamma=0.9)

        # Loss
        self.lpips_loss_fn = PerceptualLoss(net='vgg', device=device)
        self.L1_loss_fn = nn.L1Loss()
        self.ssim_loss_fn = SSIM(data_range=1, size_average=True, channel=1, nonnegative_ssim=False).to(device)

        # Save training results
        self.is_SummaryWriter = cfgs.is_SummaryWriter
        if self.is_SummaryWriter:
            self.writer = SummaryWriter('./summary/{}'\
            .format(cfgs.model_name)) 

    
    def run_train(self, cfgs):
        for epoch in range(cfgs.load_epoch_for_train, cfgs.epochs):
            lr = self.scheduler.get_last_lr()[0]
            print('lr:', lr)
              
            self.train_many_to_one(epoch, cfgs)
            self.scheduler.step()
            
            torch.save({'epoch': epoch, 'state_dict': self.model.state_dict()}, 
                        os.path.join(self.path_to_model, '{}_{}.pth.tar'\
                            .format(self.model_name, epoch+1)))                                            


    def train_many_to_one(self, epoch, cfgs):
        torch.cuda.empty_cache()
        self.model.train()
        batch_num =len(self.train_loader)
        loss = 0
        prev_img = None
        state = None
        for batch_idx, train_data in enumerate(self.train_loader):
            seq_event_patch, img_patch, gt_img_patch = train_data
            img_patch = img_patch.to(self.device)
            gt_img_patch = gt_img_patch.to(self.device)
            loss = 0
            state = None
            output = None
            prev_img = None
            # prevprev_img = None
           
            for s in range(len(seq_event_patch)):
                event_patch = seq_event_patch[s].to(self.device)

                if s == 0:
                    prev_img = torch.zeros_like(img_patch)  
                    state = None       
                output, state = self.model(event_patch, prev_img, state)
                prev_img = output.clone()
                
            # if cfgs.display_train:
            #     show_whole_img(event_patch, output, gt_img_patch)                

            loss_lpips = self.lpips_loss_fn(output, gt_img_patch,normalize=True)
            loss_mse = self.L1_loss_fn(output, gt_img_patch)
            loss_ssim = 1 - self.ssim_loss_fn(output, gt_img_patch)
            loss = loss_lpips + loss_mse + loss_ssim

            if self.is_SummaryWriter:
                self.writer.add_scalar('LPIPS', loss_lpips, batch_num*epoch+batch_idx)
                self.writer.add_scalar('MSE', loss_mse, batch_num*epoch+batch_idx)
                self.writer.add_scalar('SSIM', loss_ssim, batch_num*epoch+batch_idx)
                self.writer.add_scalar('loss', loss, batch_num*epoch+batch_idx)

            self.optimizer.zero_grad()
            loss.backward(retain_graph=False) 
            self.optimizer.step() 


            if batch_idx%50 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tloss: {:.6f}'.format(\
                    epoch+1, batch_idx*self.train_loader.batch_size, len(self.train_loader.dataset),\
                    100.*batch_idx/len(self.train_loader), loss.data)) # .data.cpu().numpy()

        self.optimizer.zero_grad()




if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('device: ', device)

    ## config parameters
    parser = argparse.ArgumentParser(
        description='Training options')
    set_configs(parser)
    cfgs = parser.parse_args()
    cfgs.shuffle = True #if cfgs.is_recurrent else True

    model_train = Train(cfgs, device)
    model_train.run_train(cfgs)

