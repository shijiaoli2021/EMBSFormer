import configparser
import sys
sys.path.append("..")
import utils.data.functions as functions
import torch


# for EMBSFomer
def get_model_config(config_path, adj_path, device):
    config = configparser.ConfigParser()
    config.read(config_path)
    K = int(config['Training']['K'])
    embedded_dim = int(config['Training']['embedded_dim'])
    hidden_dim = int(config['Training']['hidden_dim'])
    nodes_num = int(config['Data']['nodes_num'])
    if "METR-LA" in adj_path:
        adj = functions.get_adj_matrix_METR(adj_path, nodes_num)
    else:
        adj = functions.get_adj_matrix_PEM(adj_path, nodes_num)
    d_m = functions.get_d_matrix(adj)
    L = functions.scaled_laplacian(functions.get_laplace_matrix(adj, d_m))

    cheb_polynomials = [torch.FloatTensor(i).to(device)
                        for i in functions.chebyshev_ploynomials(L, K)]

    backbones = [
        {
            "K": K,
            "in_channel": embedded_dim,
            "out_channel": 64,
            "hidden_dim": hidden_dim,
            "cheb_ploynomials": cheb_polynomials
        },
        {
             "K": K,
            "in_channel": hidden_dim,
            "out_channel": hidden_dim,
            "hidden_dim": hidden_dim,
            "cheb_ploynomials": cheb_polynomials
        }
    ]


    return backbones

