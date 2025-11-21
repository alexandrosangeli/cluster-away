import argparse
import torch
from generic_utils.utils import AUTOENCODER_SELLECTION, log_params


def get_common_parser():
    """Creates a parser containing the union of ALL arguments from all scripts."""
    parser = argparse.ArgumentParser(add_help=False) # Important: Disable help now

    # common
    parser.add_argument('--output_dir', type=str, required=False, help='Specify the path to save any output')
    parser.add_argument('--data_path', type=str, required=False, help='The path with the data')
    parser.add_argument('--pdbs', type=str, nargs='+', help='Specify one or more PDBS')
    parser.add_argument('--autoencoder', type=str, required=False, help='The autoencoder type')
    parser.add_argument('--timestamp', type=str, required=False, help='The current timestamp')
    parser.add_argument('--description', type=str, default="", help='Optional description of the current experiment')
    parser.add_argument('--request_gpu', type=int, required=False, help='Flag 0/1 whether GPU was requested')
    parser.add_argument('--verbose', type=int, required=False, help='Verbosity flag')
   
    # drifts_experiment
    parser.add_argument('--checkpoint_file', type=str, help='The checkpoint (ckpt) file for the model') 
    parser.add_argument('--num_iters', type=int, help="Number of decoding-encoding iterations")
    parser.add_argument('--grid_scale_factor', type=float, help='Scale factor to expand the grid by (0 means no additional expansion)')
    parser.add_argument('--resolution', type=int, help='Resolution as number of points generated in linspace')
    return parser

def parse_all_args(description, experiment):
    """Parses all arguments, handles help, and returns them as a clean dictionary."""
    
    main_parser = argparse.ArgumentParser(description=description)
    common_parser = get_common_parser()
    parser = argparse.ArgumentParser(description=description, parents=[common_parser])
    args_dict = vars(parser.parse_args())

    assert args_dict['request_gpu'] in [0, 1], "--request_gpu should take the value of 0 or 1"
    assert (not args_dict['request_gpu']) or (torch.cuda.is_available() == args_dict['request_gpu']), "GPU was requested but is not available"
    assert args_dict['verbose'] in [0, 1], "--verbose should take the value of 0 or 1"

    # common
    args_dict['data_path'] = args_dict['data_path'] if args_dict['data_path'][-1] != '/' else args_dict['data_path'][:-1]
    args_dict['datafiles'] = [f"{args_dict['data_path']}/{pdb}" for pdb in args_dict['pdbs']]
    args_dict['output_dir'] = args_dict['output_dir'] if args_dict['output_dir'][-1] != '/' else args_dict['output_dir'][:-1]
    args_dict['device'] = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args_dict['autoencoder_of_choice'] = AUTOENCODER_SELLECTION[args_dict['autoencoder']]
    args_dict['request_gpu'] = True if args_dict['request_gpu'] == 1 else False
    args_dict['verbose'] = True if args_dict['verbose'] == 1 else False

    # drifts_experiment
    assert not experiment.endswith("drifts_experiment.py") or args_dict['res'] > 1, "Resolution must be greater than 1 otherwise the code will fail"
    
    # model training
    # None for now

    args_dict_processed = {k : v for k, v in args_dict.items() if v is not None}
    log_params(log_path=args_dict_processed['output_dir'], experiment=experiment, **args_dict_processed)
    return args_dict_processed

