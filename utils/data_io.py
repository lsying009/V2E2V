import os
import numpy as np
import matplotlib.pyplot as plt 
from PIL import Image
import csv


def make_event_preview(events, mode='grayscale', num_bins_to_show=-1):
    # events: [1 x C x H x W] event numpy or [C x H x W]
    # num_bins_to_show: number of bins of the voxel grid to show. -1 means show all bins.
    if events.ndim == 3:
        events = np.expand_dims(events,axis=0)
    if num_bins_to_show < 0:
        sum_events = np.sum(events[0, :, :, :], axis=0)
    else:
        sum_events = np.sum(events[0, -num_bins_to_show:, :, :], axis=0)

    if mode == 'red-blue':
        # Red-blue mode
        # positive events: blue, negative events: red
        event_preview = np.zeros((sum_events.shape[0], sum_events.shape[1], 3), dtype=np.uint8)
        b = event_preview[:, :, 0]
        r = event_preview[:, :, 2]
        b[sum_events > 0] = 255
        r[sum_events < 0] = 255
    else:
        # Grayscale mode
        # normalize event image to [0, 255] for display
        m, M = -5.0, 5.0
        # M = (sum_events.max() - sum_events.min())/2
        # m = -M
        
        event_preview = np.clip((255.0 * (sum_events - m) / (M - m)), 0, 255).astype(np.uint8)
        # event_preview = np.clip((255.0 * (sum_events - sum_events.min()) / (sum_events.max() - sum_events.min())).astype(np.uint8), 0, 255)

    return event_preview


class Writer:
    def __init__(self, cfgs, model_name, dataset_name=None):
        self.output_folder = cfgs.output_folder
        if not dataset_name:
            self.dataset_name = cfgs.test_data_name
        else:
            self.dataset_name = dataset_name

        if not os.path.exists(self.output_folder):
                os.makedirs(self.output_folder)
        self.output_data_folder = os.path.join(self.output_folder, model_name, '{}'.format(self.dataset_name))

class EvalWriter(Writer):
    """
    Write evaluation results to disk.
    """

    def __init__(self, cfgs, model_name, dataset_name=None):
        super(EvalWriter, self).__init__(cfgs, model_name, dataset_name)
        self.is_write_image = cfgs.is_write_image
        print('== Eval Txt Writer ==')
        if self.is_write_image:
            if not os.path.exists(self.output_data_folder):
                os.makedirs(self.output_data_folder)
            self.output_txt_file = os.path.join(self.output_data_folder, 'result.csv')
            print('Will write evaluation result to: {}'.format(self.output_txt_file))
        else:
            print('Will not write evaluation result to disk.')

    def __call__(self, name_results, results):
        if not self.is_write_image:
            return
        with open(self.output_txt_file, 'a+', newline='') as f:
            writer = csv.writer(f, delimiter='\t')
            # mse, psnr, ssim, lpips, num_frame, N_events
            writer.writerow(name_results)
            writer.writerow(results)
        f.close()


class ErrorMapWriter(Writer):
    """
    Write error_map between reconstructed and GT images to disk.
    """

    def __init__(self, cfgs, model_name, dataset_name=None):
        super(ErrorMapWriter, self).__init__(cfgs, model_name, dataset_name)
        self.is_write_emap = cfgs.is_write_emap
        
        if not dataset_name:
            self.dataset_name = cfgs.test_data_name
        else:
            self.dataset_name = dataset_name

        print('== Error Map Writer ==')
        if self.is_write_emap:
            self.output_data_folder = os.path.join(self.output_data_folder, 'error_maps')
            if not os.path.exists(self.output_data_folder):
                os.makedirs(self.output_data_folder)
            print('Will write error maps to: {}'.format(self.output_data_folder))
        else:
            print('Will not write error maps to disk.')

    def __call__(self, img, gt_img, img_id):
        if not self.is_write_emap:
            return
        diff = img.astype(np.float32)/255.- gt_img.astype(np.float32)/255.
    
        plt.imshow(diff, cmap='RdBu', vmin=-0.5, vmax=0.5)  #coolwarm
        plt.axis('off') 
        plt.savefig(os.path.join(self.output_data_folder,
                         'frame_{:010d}.png'.format(img_id)), bbox_inches='tight')


class ImageWriter(Writer):
    """
    Utility class to write images to disk.
    """

    def __init__(self, cfgs, model_name, dataset_name=None):
        super(ImageWriter, self).__init__(cfgs, model_name, dataset_name)
        self.is_write_image = cfgs.is_write_image
        
        print('== Image Writer ==')
        if self.is_write_image:
            if not os.path.exists(self.output_data_folder):
                os.makedirs(self.output_data_folder)
            print('Will write images to: {}'.format(self.output_data_folder))
        else:
            print('Will not write images to disk.')

    def __call__(self, img, img_id):
        if not self.is_write_image:
            return
        img = Image.fromarray(np.uint8(img))
        img.save(os.path.join(self.output_data_folder,
                         'frame_{:010d}.png'.format(img_id)))


class EventWriter(Writer):
    """
    Utility class to write event images to disk.
    """

    def __init__(self, cfgs, model_name, dataset_name=None):
        super(EventWriter, self).__init__(cfgs, model_name, dataset_name)

        self.is_write_event = cfgs.is_write_event

        print('== Event Writer ==')
        if self.is_write_event:
            self.output_data_folder = os.path.join(self.output_data_folder, 'events')
            if not os.path.exists(self.output_data_folder):
                os.makedirs(self.output_data_folder)
            print('Will write event tensor to: {}'.format(self.output_data_folder))
        else:
            print('Will not write events to disk.')

    def __call__(self, img, img_id):
        if not self.is_write_event:
            return
        img = Image.fromarray(np.uint8(img))
        img.save(os.path.join(self.output_data_folder,
                         'events_{:010d}.png'.format(img_id)))
        