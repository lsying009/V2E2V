'''Some codes (ConvLayer/UpsampleConvLayer/RecurrentConvLayer/RecurrentUpSampConvLayer) 
are modified from https://github.com/uzh-rpg/rpg_e2vid'''

import torch
import torch.nn as nn
import numpy as np
from torch.nn import Parameter



def softshrink(x, lambd):
    return nn.functional.relu(x - lambd) - nn.functional.relu(-x - lambd)

def doubletanh(x, u, lambd):
    return torch.mul(u, torch.tanh(x + lambd) + torch.tanh(x - lambd))
    
def connect_cat(x1, x2):
     return torch.cat((x1, x2), axis=1)


class IstaBlock(nn.Module):
    def __init__(self, base_channels=32, kernel_size=3,\
            stride=1, padding=1, activation=None, norm=None, is_recurrent=False):
        super(IstaBlock, self).__init__()

        self.D = ConvLayer(in_channels=2*base_channels, out_channels=base_channels, kernel_size=kernel_size,\
            stride=stride, padding=padding, activation=activation, norm=norm)
        self.P = ConvLayer(in_channels=base_channels, out_channels=2*base_channels, kernel_size=kernel_size,\
            stride=stride, padding=padding, activation=activation, norm=norm)

        self.Lambda = Parameter(torch.Tensor(0.001*np.random.rand(1, 2*base_channels, 1,1)))
        if is_recurrent:
            # input gate
            self.gates = ConvLayer(in_channels=3*base_channels, out_channels=4*base_channels, kernel_size=kernel_size,\
            stride=stride, padding=padding, activation=activation, norm=norm)


class ConvLSTMZ0(nn.Module):
    def __init__(self, x_size, z_size, output_size, kernel_size):
        super(ConvLSTMZ0, self).__init__()
        '''LSTC block for sparse codes Z0'''

        self.x_size = x_size
        self.z_size = z_size
        self.output_size = output_size
        pad = kernel_size // 2

        self.gates = nn.Conv2d(self.x_size+self.z_size, 2*self.output_size, kernel_size, padding=pad, padding_mode='reflect')
        self.out_gates = nn.Conv2d(self.z_size+self.output_size, self.output_size, kernel_size, padding=pad, padding_mode='reflect')
        self.P0 = nn.Conv2d(self.x_size, self.output_size, kernel_size, padding=pad, padding_mode='reflect')
    
    def forward(self, x, z=None, prev_state=None):
        B,_,H,W = x.size()
        if z is None:
            z = torch.zeros((B, self.z_size, H, W), device=x.device)
        # inputs = connect_cat(x, z)
        gates = self.gates(connect_cat(x, z))
        in_gate, forget_gate = gates.chunk(2, 1)
        in_gate = torch.sigmoid(in_gate)
        forget_gate = torch.sigmoid(forget_gate)
        z0 = self.P0(x)

        out_gate = torch.sigmoid(self.out_gates(connect_cat(z0, z)))

        if prev_state is None:
            prev_state = torch.zeros_like(z0)
        z0 = forget_gate * prev_state + in_gate * z0
        
        output = out_gate * torch.tanh(z0)
        
        return output, z0



class ConvLSTM(nn.Module):
    """Adapted from: https://github.com/Atcold/pytorch-CortexNet/blob/master/model/ConvLSTMCell.py """

    def __init__(self, input_size, hidden_size, kernel_size):
        super(ConvLSTM, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        pad = kernel_size // 2

        # cache a tensor filled with zeros to avoid reallocating memory at each inference step if --no-recurrent is enabled
        self.zero_tensors = {}

        self.Gates = nn.Conv2d(input_size + hidden_size, 4 * hidden_size, kernel_size, padding=pad, padding_mode='reflect')

    def forward(self, input_, prev_state=None):
        
        # get batch and spatial sizes
        batch_size = input_.data.size()[0]
        spatial_size = input_.data.size()[2:]

        # generate empty prev_state, if None is provided
        if prev_state is None:
            # create the zero tensor if it has not been created already
            state_size = tuple([batch_size, self.hidden_size] + list(spatial_size))
            if state_size not in self.zero_tensors:  
                # allocate a tensor with size `spatial_size`, filled with zero (if it has not been allocated already)
                self.zero_tensors[state_size] = (
                    torch.zeros(state_size).to(input_.device),
                    torch.zeros(state_size).to(input_.device)
                )

            prev_state = self.zero_tensors[tuple(state_size)]
       
        prev_hidden, prev_cell = prev_state

        # data size is [batch, channel, height, width]
        stacked_inputs = torch.cat((input_, prev_hidden), 1) #prev_hidden
        gates = self.Gates(stacked_inputs)

        # chunk across channel dimension
        in_gate, remember_gate, out_gate, cell_gate = gates.chunk(4, 1)

        # apply sigmoid non linearity
        in_gate = torch.sigmoid(in_gate)
        remember_gate = torch.sigmoid(remember_gate)
        out_gate = torch.sigmoid(out_gate)

        # apply tanh non linearity
        cell_gate = torch.tanh(cell_gate)

        # compute current cell and hidden state
        cell = (remember_gate * prev_cell) + (in_gate * cell_gate)
        hidden = out_gate * torch.tanh(cell)
      
        return hidden, cell




class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, activation=None, norm=None):
        super(ConvLayer, self).__init__()

        bias = False if norm == 'BN' else True
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias, padding_mode='reflect')
        if activation is not None:
            self.activation = getattr(torch, activation, 'relu')
        else:
            self.activation = None

        self.norm = norm
        if norm == 'BN':
            self.norm_layer = nn.BatchNorm2d(out_channels, momentum=0.1)
        elif norm == 'IN':
            self.norm_layer = nn.InstanceNorm2d(out_channels, track_running_stats=True)

    def forward(self, x):
        out = self.conv2d(x)

        if self.norm in ['BN', 'IN']:
            out = self.norm_layer(out)

        if self.activation is not None:
            out = self.activation(out)

        return out




class UpsampleConvLayer(nn.Module):
    """
    an upsample convolution layer which includes a nearest interpolate and a general_conv2d
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, activation=None, norm=None):
        super(UpsampleConvLayer, self).__init__()
        # self._in_channels = in_channels
        # self._out_channels = out_channels
        self._kernel_size = kernel_size

        bias = False if norm == 'BN' else True
        # new_pad = self._kernel_size-1-padding
        self.pad = nn.ReflectionPad2d(padding=(int((self._kernel_size-1)/2), int((self._kernel_size-1)/2),
                                        int((self._kernel_size-1)/2), int((self._kernel_size-1)/2)))#对称padding
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)
        if activation is not None:
            self.activation = getattr(torch, activation, 'relu')
        else:
            self.activation = None

        self.norm = norm
        if norm == 'BN':
            self.norm_layer = nn.BatchNorm2d(out_channels, momentum=0.1)
        elif norm == 'IN':
            self.norm_layer = nn.InstanceNorm2d(out_channels, track_running_stats=True)


    def forward(self, conv, out_dim=None):
        shape = conv.shape
        # out_dim = (2*shape[2]-1, 2*shape[3]-1)
    
        if out_dim is None:
            conv = nn.functional.interpolate(conv,size=[shape[2]*2,shape[3]*2],mode='bilinear', align_corners=False)#最近邻插值上采样
        else:
            conv = nn.functional.interpolate(conv,size=[out_dim[0],out_dim[1]],mode='bilinear', align_corners=False)
        out = self.pad(conv)
        out = self.conv2d(out)

        if self.norm in ['BN', 'IN']:
            out = self.norm_layer(out)

        if self.activation is not None:
            out = self.activation(out)
        
        return out



class RecurrentConvLayer(nn.Module):
    def __init__(self,in_channels, out_channels, kernel_size=3, stride=1, padding=0,
                 activation=None, norm=None):
        super(RecurrentConvLayer, self).__init__()
        self.conv = ConvLayer(in_channels, out_channels, kernel_size, stride, padding, activation, norm)
        self.recurrent_block = ConvLSTM(input_size=out_channels, hidden_size=out_channels, kernel_size=3)

    def forward(self, x, prev_state):
        x = self.conv(x)
        state = self.recurrent_block(x, prev_state)
        x = state[0]
        return x, state


class RecurrentUpSampConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0, activation=None, norm=None):
        super(RecurrentUpSampConvLayer, self).__init__()
        self.upsample_conv2d = UpsampleConvLayer(in_channels, out_channels, kernel_size, stride, padding, activation, norm)
        self.recurrent_block = ConvLSTM(input_size=out_channels, hidden_size=out_channels, kernel_size=3)

    def forward(self, x, prev_state, out_dim=None):
        x = self.upsample_conv2d(x, out_dim=out_dim)
        state = self.recurrent_block(x, prev_state)
        x = state[0]
        return x, state


