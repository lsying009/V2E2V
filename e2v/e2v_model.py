import torch.nn as nn
from .base_layers import *


class CistaLSTCNet(nn.Module):
     def __init__(self, image_dim, base_channels=64, depth=5, num_bins=5):
          super(CistaLSTCNet, self).__init__()
          '''
               CISTA-LSTC network for events-to-video reconstruction
          '''
          self.num_bins = num_bins
          self.depth = depth
          self.height, self.width = image_dim
          self.num_states = 3
          

          self.We = ConvLayer(in_channels=self.num_bins, out_channels=int(base_channels/2), kernel_size=3,\
          stride=1, padding=1) #We_new 
          self.Wi = ConvLayer(in_channels=1, out_channels=int(base_channels/2), kernel_size=3,\
               stride=1, padding=1) 
          self.W0 = ConvLayer(in_channels=base_channels, out_channels=base_channels, kernel_size=3,\
               stride=2, padding=1) 

          self.P0 = ConvLSTMZ0(x_size=base_channels, z_size=2*base_channels, output_size=2*base_channels, kernel_size=3) 

          lista_block = IstaBlock(base_channels=base_channels, is_recurrent=False) 
          self.lista_blocks = nn.ModuleList([lista_block for i in range(self.depth)])
          

          self.Dg = RecurrentConvLayer(in_channels=2*base_channels, out_channels=base_channels, kernel_size=3, stride=1, padding=1,
               activation='relu')

          self.upsamp_conv = UpsampleConvLayer(in_channels=base_channels, out_channels=base_channels, kernel_size=3, stride=1, padding=0, activation='relu')
          
          self.final_conv = ConvLayer(in_channels=base_channels, out_channels=1, \
               kernel_size=3, stride=1, padding=1)
          
          self.sigmoid = nn.Sigmoid()


     def forward(self, events, prev_image, prev_states):
          '''
          Inputs:
               events: torch.tensor, float32, [batch_size, num_bins, H, W]
                    Event voxel grid
               prev_image: torch.tensor, float32, [batch_size, 1, H, W]
                    Reconstructed frame from the last reconstruction
               prev_states: None or list of torch.tensor, float32
                    Previous states
          Outputs:
               rec_I: torch.tensor, float32, [batch_size, 1, H, W]
                    Reconstructed frame
               states: list of torch.tensor, float32
                    Updated states in e2v_net
          '''
          
          if prev_states is None:
               prev_states = [None]*self.num_states
          states = [] 

          
          x_E = self.We(events)
          x_I = self.Wi(prev_image)
          x1 = connect_cat(x_E, x_I) 

          x1 = self.W0(x1) 

          z, state = self.P0(x1, prev_states[-2], prev_states[0] if prev_states[0] is not None else None)
          states.append(state)
          tmp = z.clone()

          for i in range(self.depth):
               tmp = self.lista_blocks[i].D(tmp)
               x = x1- tmp
               x = self.lista_blocks[i].P(x)
               x = x + z  # + temporal_z
               z = softshrink(x, self.lista_blocks[i].Lambda) 
               tmp = z      

          states.append(z)
          
          rec_I, state = self.Dg(z, prev_states[-1])
          states.append(state)

          rec_I = self.upsamp_conv(rec_I)

          rec_I = self.final_conv(rec_I)
          rec_I = self.sigmoid(rec_I)

          return rec_I, states


class CistaTCNet(nn.Module):
     def __init__(self, base_channels=32, depth=5, num_bins=5):
          super(CistaTCNet, self).__init__()
          '''
               CISTA-TC network for events-to-video reconstruction
          '''
          self.num_bins = num_bins
          self.depth = depth
          self.num_states = 2 

          self.one_conv_for_prev = ConvLayer(in_channels=2*base_channels, out_channels=1, \
               kernel_size=3, stride=1, padding=1)
          self.one_conv_for_cur = ConvLayer(in_channels=2*base_channels, out_channels=1, \
               kernel_size=3, stride=1, padding=1)
          alpha = nn.Parameter(torch.Tensor([0.001*np.random.rand(2*base_channels, 1,1)]))
          self.alpha = nn.ParameterList([ alpha for i in range(self.depth)])

               
          self.We = ConvLayer(in_channels=self.num_bins, out_channels=int(base_channels/2), kernel_size=3,\
          stride=1, padding=1)
          self.Wi = ConvLayer(in_channels=1, out_channels=int(base_channels/2), kernel_size=3,\
               stride=1, padding=1) 
          self.W0 = ConvLayer(in_channels=base_channels, out_channels=base_channels, kernel_size=3,\
               stride=2, padding=1)

          self.P0 = ConvLayer(in_channels=base_channels, out_channels=2*base_channels, kernel_size=3,\
               stride=1, padding=1)#64

          lista_block = IstaBlock(base_channels=base_channels, is_recurrent=False)
          self.lista_blocks = nn.ModuleList([lista_block for i in range(self.depth)])

          self.Dg = RecurrentConvLayer(in_channels=2*base_channels, out_channels=base_channels, kernel_size=3, stride=1, padding=1,
                    activation='relu') 

          self.upsamp_conv = UpsampleConvLayer(in_channels=base_channels, out_channels=base_channels, kernel_size=3, stride=1, padding=0, activation=None, norm=None)
          self.final_conv = ConvLayer(in_channels=base_channels, out_channels=1, \
               kernel_size=3, stride=1, padding=1)
          
          self.sigmoid = nn.Sigmoid()
     def calc_attention_feature(self, img1, img2, prev_attention_state):# , prev_attention_state):
          # TSA
          S1 = self.sim_layers(img1)
          S2 = self.sim_layers(img2)
          feat1 =  self.one_conv1(S1)
          feat2 = self.one_conv2(S2)
          attention_map = torch.sigmoid(torch.mul(feat1,feat2)) #attention state
          # return attention_map
          if prev_attention_state is None:
               prev_attention_state = torch.ones_like(attention_map)
          attention1 = torch.mul(S1, prev_attention_state)
          attention2 = torch.mul(S2, attention_map)
          return attention1, attention2, attention_map

     def forward(self, events, prev_img, prev_states):
          '''
          Inputs:
               events: torch.tensor, float32, [batch_size, num_bins, H, W]
                    Event voxel grid
               prev_img: torch.tensor, float32, [batch_size, 1, H, W]
                    Reconstructed frame from the last reconstruction
               prev_states: None or list of torch.tensor, float32
                    Previous states
          Outputs:
               rec_I: torch.tensor, float32, [batch_size, 1, H, W]
                    Reconstructed frame
               states: list of torch.tensor, float32
                    Updated states in e2v_net
          '''
          # input event tensor Ek, Ik-1, Ik,
          if prev_states is None:
               prev_states = [None]*self.num_states
          states = [] 


          x_E = self.We(events)
          x_I = self.Wi(prev_img)

          x1 = self.W0(connect_cat(x_E, x_I) ) 
          z = self.P0(x1)
          tmp = z
          if prev_states[0] is None:
              prev_states[0] = torch.zeros_like(z)
          
          one_ch_prev_z = self.one_conv_for_prev(prev_states[0])
          for i in range(self.depth):
               one_ch_cur_z = self.one_conv_for_cur(tmp)
               attention_map = torch.sigmoid(torch.mul(one_ch_prev_z, one_ch_cur_z))
               temporal_z = attention_map*torch.mul((prev_states[0]-tmp), self.alpha[i])
               tmp = self.lista_blocks[i].D(tmp)
               x = x1- tmp
               x = self.lista_blocks[i].P(x)
               x = x + z + temporal_z
               z = softshrink(x, self.lista_blocks[i].Lambda) 
               tmp = z      

          states.append(z)
          
          rec_I, state = self.Dg(z, prev_states[-1])
          states.append(state)

          rec_I = self.upsamp_conv(rec_I)
          rec_I = self.final_conv(rec_I)
          rec_I = self.sigmoid(rec_I)
          
          return rec_I, states 


