'''Event processing, modified from
https://github.com/uzh-rpg/rpg_e2vid/blob/master/utils/inference_utils.py
'''

import numpy as np
import torch

def add_noise_to_voxel(voxel, noise_std=1.0, noise_fraction=0.1):
    noise = noise_std * torch.randn_like(voxel)  # mean = 0, std = noise_std
    if noise_fraction < 1.0:
        mask = torch.rand_like(voxel) >= noise_fraction
        noise.masked_fill_(mask, 0)
    return voxel + noise

def events_to_voxel_grid(events, num_bins, width, height):
    """
    Build a voxel grid with bilinear interpolation in the time domain from a set of events.
    :param events: a [N x 4] NumPy array containing one event per row in the form: [timestamp, x, y, polarity]
    :param num_bins: number of bins in the temporal axis of the voxel grid
    :param width, height: dimensions of the voxel grid
    """

    assert(events.shape[1] == 4)
    assert(num_bins > 0)
    assert(width > 0)
    assert(height > 0)

    voxel_grid = np.zeros((num_bins, height, width), np.float32).ravel()

    if len(events) == 0:
        return np.reshape(voxel_grid, (num_bins, height, width))
    # normalize the event timestamps so that they lie between 0 and num_bins
    last_stamp = events[-1, 0]
    first_stamp = events[0, 0]
    deltaT = last_stamp - first_stamp

    if deltaT == 0:
        deltaT = 1.0

    events[:, 0] = (num_bins - 1) * (events[:, 0] - first_stamp) / deltaT
    ts = events[:, 0]
    xs = events[:, 1].astype(np.uint)
    ys = events[:, 2].astype(np.uint)
    pols = events[:, 3]
    pols[pols == 0] = -1  # polarity should be +1 / -1

    tis = ts.astype(np.uint)
    dts = ts - tis
    vals_left = pols * (1.0 - dts)
    vals_right = pols * dts

    valid_indices = tis < num_bins
    np.add.at(voxel_grid, xs[valid_indices] + ys[valid_indices] * width
              + tis[valid_indices] * width * height, vals_left[valid_indices])

    valid_indices = (tis + 1) < num_bins
    np.add.at(voxel_grid, xs[valid_indices] + ys[valid_indices] * width
              + (tis[valid_indices] + 1) * width * height, vals_right[valid_indices])

    voxel_grid = np.reshape(voxel_grid, (num_bins, height, width))


    return voxel_grid


def events_to_voxel_grid_pytorch(events, num_bins, width, height):#, divide_sign=True):
    """
    Build a voxel grid with bilinear interpolation in the time domain from a set of events.
    :param events: a [N x 4] NumPy array containing one event per row in the form: [timestamp, x, y, polarity]
    :param num_bins: number of bins in the temporal axis of the voxel grid
    :divide_sign: True, neg/pos divide into two groups
    :param width, height: dimensions of the voxel grid
    :param device: device to use to perform computations
    :return voxel_grid: PyTorch event tensor (on the device specified)
    """


    assert(events.shape[1] == 4)
    assert(num_bins > 0)
    assert(width > 0)
    assert(height > 0)

    with torch.no_grad():
    #events_torch = torch.from_numpy(events)
    #events_torch = events_torch.to(device)
        voxel_grid = torch.zeros(num_bins, height, width, dtype=torch.float32, device=events.device).flatten()
        
        if len(events) == 0:
            return voxel_grid.view(num_bins, height, width)
        # normalize the event timestamps so that they lie between 0 and num_bins
        last_stamp = events[-1, 0]
        first_stamp = events[0, 0]
        deltaT = last_stamp - first_stamp

        if deltaT == 0:
            deltaT = 1.0

        events[:, 0] = (num_bins - 1) * (events[:, 0] - first_stamp) / deltaT
        ts = events[:, 0]
        xs = events[:, 1].long()
        ys = events[:, 2].long()
        pols = events[:, 3].float()
        pols[pols == 0] = -1  # polarity should be +1 / -1

        tis = torch.floor(ts)
        tis_long = tis.long()
        dts = ts - tis
        vals_left = pols * (1.0 - dts.float())
        vals_right = pols * dts.float()

        valid_indices = tis < num_bins
        valid_indices &= tis >= 0
        voxel_grid.index_add_(dim=0,
                            index=xs[valid_indices] + ys[valid_indices]
                            * width + tis_long[valid_indices] * width * height,
                            source=vals_left[valid_indices])

        valid_indices = (tis + 1) < num_bins
        valid_indices &= tis >= 0

        voxel_grid.index_add_(dim=0,
                            index=xs[valid_indices] + ys[valid_indices] * width
                            + (tis_long[valid_indices] + 1) * width * height,
                            source=vals_right[valid_indices])


    voxel_grid = voxel_grid.view(num_bins, height, width)

    return voxel_grid


def event_preprocess(event_voxel_grid, mode='std', filter_hot_pixel=False):
# Normalize the event tensor (voxel grid) so that
# the mean and stddev of the nonzero values in the tensor are equal to (0.0, 1.0)
    num_bins = event_voxel_grid.shape[0]
    if filter_hot_pixel:
        event_voxel_grid[abs(event_voxel_grid) > 25./num_bins]= 0

    if mode == 'maxmin':
        event_voxel_grid = (event_voxel_grid- event_voxel_grid.min())/(event_voxel_grid.max()- event_voxel_grid.min()+1e-8)
    elif mode == 'std':
        nonzero_ev = (event_voxel_grid != 0)
        num_nonzeros = nonzero_ev.sum()
        if num_nonzeros > 0:
            # compute mean and stddev of the **nonzero** elements of the event tensor
            # we do not use PyTorch's default mean() and std() functions since it's faster
            # to compute it by hand than applying those funcs to a masked array
            mean = event_voxel_grid.sum() / num_nonzeros
            mask = nonzero_ev.astype(np.float32)
            stddev = np.sqrt((event_voxel_grid ** 2).sum() / num_nonzeros - mean ** 2)
            
            event_voxel_grid = mask * (event_voxel_grid - mean) / (stddev + 1e-8)

    return event_voxel_grid


def event_preprocess_pytorch(event_voxel_grid, mode='std', filter_hot_pixel=True):
# Normalize the event tensor (voxel grid) so that
# the mean and stddev of the nonzero values in the tensor are equal to (0.0, 1.0)
    num_bins = event_voxel_grid.shape[0]
    if filter_hot_pixel:
        event_voxel_grid[abs(event_voxel_grid) > 20./num_bins]= 0
    if mode == 'maxmin':
        event_voxel_grid = (event_voxel_grid- event_voxel_grid.min())/(event_voxel_grid.max()- event_voxel_grid.min()+1e-8)
    elif mode == 'std':
        nonzero_ev = (event_voxel_grid != 0)
        num_nonzeros = nonzero_ev.sum()
        if num_nonzeros > 0:
            # compute mean and stddev of the **nonzero** elements of the event tensor
            # we do not use PyTorch's default mean() and std() functions since it's faster
            # to compute it by hand than applying those funcs to a masked array
            mean = event_voxel_grid.sum() / num_nonzeros
            mask = nonzero_ev.float()
            stddev = torch.sqrt((event_voxel_grid ** 2).sum() / num_nonzeros - mean ** 2)                
            event_voxel_grid = mask * (event_voxel_grid - mean) / (stddev+1e-8)

    return event_voxel_grid


