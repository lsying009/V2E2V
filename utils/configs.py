import argparse
import os

def set_configs(parser):
    ##------------------------ basic info -------------------------##
    parser.add_argument('--image_dim', nargs=2, default=[180, 240], type=int,
                        help='The height and width of input frames (height, width) (default:[180,240])')
    parser.add_argument('--path_to_model', default='models', type=str,
                        help='Path to save models (default: models)')
    parser.add_argument('--path_to_test_model', type=str,
                        help='The complete path to the model for inference')
    parser.add_argument('--model_name', default='', type=str, 
                        help='model name to save (the initial part)')
    
    ##--------------------- E2V network parameters ----------------------#
    parser.add_argument('--model_mode', default='cista-lstc', type=str,
                        help='The model type cista-lstc or cista-tc')
    parser.add_argument('-b', '--num_bins', default=5, type=int,
                        help='number of bins for event voxel grid (default: 5)')
    parser.add_argument('-d', '--depth', default=5, type=int,
                        help='The number of ISTA blocks (default: 5)') 
    parser.add_argument('-c', '--base_channels', default=64, type=int,
                        help='The base number of channels') 
    
    
    ##----------------------- V2E model parameters ----------------------#
    parser.add_argument('--num_pack_frames', default=10, type=int,
                        help='The frame count contained in a pack of frames (HFR) for each reconstruction')
    parser.add_argument('--event_mode', default='voxel_grid', type=str,
                        help='output event mode of v2e model, raw or voxel_grid, (default: voxel_grid for reconstruction)')
    parser.add_argument('--refractory_period_s', default=0.001, type=float,
                        help='Refractory period (in seconds) (default: 0.001)')
    parser.add_argument('--C', default=0.6, type=float,
                        help='Nominal contrast threshold for positive/negative events')
    parser.add_argument('--threshold_sigma', default=0.03, type=float,
                        help='Standard deviation for contrast threshold (default:0.03)')
    parser.add_argument('--cutoff_hz', default=0, type=float,
                        help='Cutoff frequency (Hz) of IIR low pass filtering (default: 0, no filtering)')
    parser.add_argument('--ps', default=1, type=float,
                        help='Coefficeint of contrast threshold for a small portion of pixels (default: 1.0), Cs = ps*C') 
    parser.add_argument('--pl', default=1, type=float,
                        help='Coefficeint of contrast threshold for a large portion of pixels (default: 1.0), Cl = pl*C')
    parser.add_argument('--qs', default=1, type=float,
                        help='Coefficeint of cutoff frequency for a small portion of pixels (default: 1.0), fc_s = qs*C')
    parser.add_argument('--ql', default=1, type=float,
                        help='Coefficeint of cutoff frequency for a large portion of pixels (default: 1.0), fc_l = ql*C')
    
    
    ##------------------------- Training --------------------------##
    parser.add_argument('--path_to_e2v', type=str, 
                        help='Path to the pretrained e2v model (trained with normal events)') 
    parser.add_argument('--path_to_train_data', type=str,
                        help='Path to training dataset')
    parser.add_argument('-s', '--len_sequence', default=10, type=int,
                        help='Length of sequence: the number of reconstructions for each loss computation')
    parser.add_argument('--no_shuffle', dest='shuffle', action='store_false',
                        help='Not shuffle data if use')
    parser.set_defaults(shuffle=True)
    parser.add_argument('--epochs', default=1, type=int)
    parser.add_argument('--lr', default=0.0001, type=float,
                        help='Learning rate')
    parser.add_argument('--batch_size', default=1, type=int,
                        help='Batch size (default: 1), if the length of sequence is not fixed, it must be 1')
    parser.add_argument('--is_SummaryWriter', dest='is_SummaryWriter', action='store_true',
                        help='Whether to save training details using summarywriter (default:False)')
    parser.set_defaults(is_SummaryWriter=False)
    parser.add_argument('--load_epoch_for_train', default=0, type=int,
                        help='Load epoch for continue training')
    parser.add_argument('--load_lr', default=0.0001, type=float,
                        help='load learning rate for continue training')
    ## for e2v training
    parser.add_argument('--add_noise', dest='add_noise', action='store_true')
    parser.set_defaults(add_noise=False)
    
    
    ##--------------------------- Testing -------------------------##
    # test dataset info
    parser.add_argument('--path_to_test_data', type=str,
                        help='Path to test dataset')
    parser.add_argument('--reader_type', default='image_reader', type=str,
                        help='Types of data loader, upsampling/image_reader/video')
    parser.add_argument('--test_data_name', default=None, type=str,
                        help='The folder name of a sequence in the test dataset, \
                        if None, test all the sequences')
    parser.add_argument('--time_unit', default='s', type=str,
                        help='Time unit in the timestamps.txt, s/ns/ms,(default:s)')
    parser.add_argument('--test_img_num', default=50, type=int,
                        help='The max number of images to test in each sequence')

    ## for e2v testing
    parser.add_argument('--num_events', default=15000, type=int,
                        help='The limited number of events per reconstruction (default: 15000)')
    parser.add_argument('--test_data_mode', default='real', type=str,
                        help='Test data type, real or upsampled') #or simulate
    
    
    # save info
    parser.add_argument('-o', '--output_folder', default='test_images', type=str,
                        help='Folder to save the inference results (default: test_images)')
    parser.add_argument('--no_write_image', dest='is_write_image', action='store_false',
                        help='Whether not to save reconstructed images (default:False, write image as default)')
    parser.set_defaults(is_write_image=True)
    parser.add_argument('--is_write_event', dest='is_write_event', action='store_true',
                        help='Whether to save event images (default: False)')
    parser.set_defaults(is_write_event=False)
    parser.add_argument('--is_write_emap', dest='is_write_emap', action='store_true',
                        help='Whether to save error maps between reconstructed and GT images (default:False)')
    parser.set_defaults(is_write_emap=False)



    # image display options
    parser.add_argument( '--display_train', dest='display_train', action='store_true')
    parser.set_defaults(display_train=False)
    parser.add_argument('--display_test', dest='display_test', action='store_true')
    parser.set_defaults(display_test=False)
    parser.add_argument( '--show_events', dest='show_events', action='store_true')
    parser.set_defaults(show_events=True)
    parser.add_argument( '--event_display_mode', default='grayscale', type=str)
    parser.add_argument( '--num_bins_to_show', default=-1, type=int)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='testing options')
    set_configs(parser)

    args = parser.parse_args()
    print(args.train_mode)
    print(args.shuffle)



