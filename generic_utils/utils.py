import molearn
from molearn.models.CNN_autoencoder import AutoEncoder as ConvolutionalAE
from molearn.models.foldingnet import AutoEncoder as FoldingNet
import json


AUTOENCODER_SELLECTION = {
    "cnn_ae" : ConvolutionalAE,
    "fold_net" : FoldingNet
}

AUTOENCODER_DEFAULT_MANDATORY_ARGUMENTS = {
    "cnn_ae" : {},
    "fold_net" : {"out_points" : 2145}
}


def log_params(path, **params):
    print("--- Parameters ---")
    print(json.dumps(params, indent=4))
    print("------------------")
    filename = f"{path}/params.json"
    with open(filename, 'w') as f:
            json.dump(params, f, indent=4)
    print(f"Parameters successfully written to: {filename}")