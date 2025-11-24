import math
import os
import sys
import time
import torch
from molearn.analysis import MolearnAnalysis

current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.join(current_dir, '..', '..') 
sys.path.append(root_dir)
from generic_utils.utils import AUTOENCODER_SELLECTION, get_data
from generic_utils.cli_utils import parse_all_args


def main(args):
    device = args['device']
    training_datafiles = args['datafiles']
    autoencoder_of_choice = args['autoencoder_of_choice']
    timestamp = args['timestamp']
    request_gpu = args['request_gpu']
    description = args['description']
    verbose = args['verbose']
    output_dir = args['output_dir']
    

    # Experiment specific
    checkpoint_file = args['checkpoint_file']

    checkpoint = torch.load(checkpoint_file, map_location=torch.device('cpu'), weights_only=False)
    model = autoencoder_of_choice(**checkpoint['network_kwargs'])
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    model.to(device)

    data = get_data(training_datafiles)
    MA = MolearnAnalysis()
    MA.set_network(model)
    MA.set_dataset('training', data)

    # Starting analysis ops
    err = MA.get_error('training')

    print(err)
    




if __name__ == "__main__":
    start_time = time.time()
    args = parse_all_args(description="Analysis arg parser", experiment=sys.argv[0])
    main(args)
    end_time = time.time()
    duration_seconds = end_time - start_time
    minutes = math.floor(duration_seconds / 60) 
    remaining_seconds = duration_seconds % 60
    print(f"Python script duration: **{minutes} minutes and {remaining_seconds:.2f} seconds**")