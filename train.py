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

import numpy as np
import os
import torch
import torch.utils.data as data
from torch import optim, nn 
from tensorboardX import SummaryWriter
from data_readers.train_data_loaders import TrainSeqData
from utils.evaluate import PerceptualLoss
from pytorch_msssim import SSIM
from utils.configs import set_configs
import argparse
from model_v2e2v import V2E2VNet
import matplotlib.pyplot as plt


class Train:
    def __init__(self, cfgs, device): 
        self.device = device 
        
        self.model_name = '{}_C{}_{}_{}_fc{}_{}_{}'\
            .format(cfgs.model_name, cfgs.C, cfgs.pl, cfgs.ps, cfgs.cutoff_hz, cfgs.ql, cfgs.qs)
        self.path_to_model = os.path.join(cfgs.path_to_model, self.model_name)
        if not os.path.exists(self.path_to_model):
            os.makedirs(self.path_to_model)
        
        self.model = V2E2VNet(cfgs=cfgs, image_dim=cfgs.image_dim, device=device).to(device)
        print(self.model) 
        
        self.v2e_params = {'C': cfgs.C,
            'ps': cfgs.ps,
            'pl': cfgs.pl,
            'cutoff_hz': cfgs.cutoff_hz,
            'qs': cfgs.qs,
            'ql': cfgs.ql,
            'refractory_period_s': cfgs.refractory_period_s
            }
        
        
        if cfgs.load_epoch_for_train:
            ## load trained E2V model using the new V2E generator
            checkpoint = torch.load(os.path.join(self.path_to_model, '{}_{}.pth.tar'
                .format(self.model_name, cfgs.load_epoch_for_train)))
            self.model.load_state_dict(checkpoint['state_dict'], strict=True)
        else:
            ## load pretrained E2V model using normal events
            checkpoint = torch.load(cfgs.path_to_e2v, map_location='cuda:0')
            self.model.e2v_net.load_state_dict(checkpoint['state_dict'], strict=True)

        # Training data
        path_to_train_data = cfgs.path_to_train_data
        train_data = TrainSeqData(os.path.join(path_to_train_data, 'train_v2e2v.txt'),  cfgs.path_to_train_data, cfgs.len_sequence, cfgs.num_pack_frames) 
        self.train_loader = data.DataLoader(train_data,batch_size=cfgs.batch_size,shuffle=cfgs.shuffle,num_workers=4)       

        # Training details
        lr = cfgs.lr*(0.9**np.floor(cfgs.load_epoch_for_train/10.)) 
        self.optimizer = optim.Adam(self.model.parameters(),lr=lr) #Adam
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=10, gamma=0.9)

        # Loss
        self.lpips_loss_fn = PerceptualLoss(net='vgg', device=device)
        self.L1_loss_fn = nn.L1Loss().to(device)
        self.ssim_loss_fn = SSIM(data_range=1, size_average=True, channel=1, nonnegative_ssim=False).to(device)

        # Save training results
        self.is_SummaryWriter = cfgs.is_SummaryWriter
        if self.is_SummaryWriter:
            self.writer = SummaryWriter('./summary/{}'\
            .format(self.model_name))
    

    def run_train(self, cfgs):
        for epoch in range(cfgs.load_epoch_for_train, cfgs.epochs):
            lr = self.scheduler.get_last_lr()[0]
            print('lr:', lr)
            
            self.train_recurrent(epoch, cfgs)
            self.scheduler.step()  
            

            torch.save({'epoch': epoch+1, 
                        'state_dict': self.model.state_dict(),
                        'v2e_params': self.v2e_params},
                        os.path.join(self.path_to_model, '{}_{}.pth.tar'\
                            .format(self.model_name, epoch+1)))   
                                  
    

    def train_recurrent(self, epoch, cfgs):
        torch.cuda.empty_cache()
        self.model.train()
        batch_num =len(self.train_loader)
        loss = 0
        prev_img = None
        state = None
        for batch_idx, train_data in enumerate(self.train_loader):
            seq_timestamps, seq_images, seq_gt_images = train_data
            loss = 0
            state = None
            
            for s in range(len(seq_timestamps)):
                timestamps = seq_timestamps[s].to(self.device)
                images = seq_images[s].to(self.device)
                gt_images = seq_gt_images[s].to(self.device)
                
                if s == 0:
                    loss = 0
                    prev_img = torch.zeros_like(gt_images)
                    state = None

                output, state= \
                    self.model(images, timestamps, prev_img, state, batch_idx)
                output = torch.clamp(output, min=1e-7, max=1-1e-7)
                prev_img = output.clone()
                
            loss_lpips = self.lpips_loss_fn(output, gt_images, normalize=True)
            loss_l1 = self.L1_loss_fn(output, gt_images)
            loss_ssim = 1 - self.ssim_loss_fn(output, gt_images)
            loss = loss_lpips + loss_l1 + loss_ssim 
            #loss
            if self.is_SummaryWriter:
                self.writer.add_scalar('LPIPS', loss_lpips, batch_num*epoch+batch_idx)
                self.writer.add_scalar('MSE', loss_l1, batch_num*epoch+batch_idx)
                self.writer.add_scalar('SSIM', loss_ssim, batch_num*epoch+batch_idx)
                self.writer.add_scalar('loss', loss, batch_num*epoch+batch_idx)

            if cfgs.display_train:
                # for name, parms in self.model.named_parameters():	
                #     print('-->name:', name, '-->grad_requirs:',parms.requires_grad)#, \
                        #' -->grad_value:',parms.grad, '-->value', parms.data)  
                plt.subplot(1,2,1)
                plt.imshow(gt_images.cpu().data[0,0,...], cmap='gray')
                plt.axis('off')
                plt.title('GT')
                plt.subplot(1,2,2)
                plt.imshow(output.cpu().data[0,0,...], cmap='gray')
                plt.axis('off')
                plt.title('I_rec')
                plt.show()

            self.optimizer.zero_grad()
            loss.backward(retain_graph=False) 
            self.optimizer.step()  
 
            if batch_idx%50 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tloss: {:.6f}'.format(\
                    epoch+1, batch_idx*self.train_loader.batch_size, len(self.train_loader.dataset),\
                    100.*batch_idx/len(self.train_loader), loss.data)) # .data.cpu().numpy()




if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('device: ', device)

    ## Parameter configuration
    parser = argparse.ArgumentParser(
        description='Training options')
    set_configs(parser)
    cfgs = parser.parse_args()
    cfgs.shuffle = True

    # Training
    model_train = Train(cfgs, device)
    model_train.run_train(cfgs)

