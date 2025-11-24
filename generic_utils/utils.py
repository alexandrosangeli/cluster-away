import molearn
from molearn.models.CNN_autoencoder import AutoEncoder as ConvolutionalAE
from molearn.models.foldingnet import AutoEncoder as FoldingNet
from molearn.data import PDBData
import json


class __ParamsEncoder(json.JSONEncoder):
    """
    Custom JSONEncoder to handle objects (e.g. torch.device) 
    """
    def default(self, obj):
        try:
            return super().default(obj)
        except TypeError:
            return str(obj)

def log_params(**params):
    print("--- Parameters ---")
    log_path = params['log_path']
    print(json.dumps(params, indent=4, cls=__ParamsEncoder))
    print("------------------")
    filename = f"{log_path}/params.json"
    with open(filename, 'w') as f:
            json.dump(params, f, indent=4, cls=__ParamsEncoder)
    print(f"Parameters successfully written to: {filename}")

def get_data(datafiles, atoms_select=True):
    print("Loading data...")
    data = PDBData()
    data.import_pdb(datafiles)
    if atoms_select:
        data.atomselect(atoms=['CA', 'C', 'CB', 'N', 'O'])
    data.prepare_dataset()
    print("Data loaded.")
    return data

AUTOENCODER_SELLECTION = {
    "cnn_ae" : ConvolutionalAE,
    "fold_net" : FoldingNet
}

AUTOENCODER_DEFAULT_MANDATORY_ARGUMENTS = {
    "cnn_ae" : {},
    "fold_net" : {"out_points" : 2145}
}

