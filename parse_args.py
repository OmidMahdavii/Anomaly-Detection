import argparse
import torch

def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument('--experiment', type=str, default='gan', choices=['gan', 'autoencoder'])
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate.')
    parser.add_argument('--max_iterations', type=int, default=5000, help='Number of training iterations.')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--window_size', type=int, default=50)
    parser.add_argument('--latent_size', type=int, default=20)
    parser.add_argument('--reg_weight', type=float, default=1.0, help='Loss regularization weight.')
    parser.add_argument('--threshold', type=float, default=0.5, help='Anomaly score threshold.')
    parser.add_argument('--print_every', type=int, default=50)
    parser.add_argument('--validate_every', type=int, default=100)
    parser.add_argument('--output_path', type=str, default='.', help='Where to create the output directory containing logs and weights.')
    parser.add_argument('--cpu', action='store_true', help='If set, the experiment will run on the CPU.')
    parser.add_argument('--test', action='store_true', help='If set, only the evaluation stage is performed.')
    parser.add_argument('--validate', action='store_true', help='If set, only the validation stage is performed.')
    
    # Additional arguments can go below this line:
    #parser.add_argument('--test', type=str, default='some default value', help='some hint that describes the effect')

    # Build options dict
    opt = vars(parser.parse_args())

    if not opt['cpu']:
        assert torch.cuda.is_available(), 'You need a CUDA capable device in order to run this experiment. See `--cpu` flag.'

    opt['output_path'] = f'{opt["output_path"]}/record/{opt["experiment"]}'

    return opt