from dataset import Arithmetic
from models import Base
from train import *
from hyperparam import search_space, top_n_params, train_custom
import torch
import sys

if torch.backends.mps.is_available():
    print("MPS")
    device = torch.device('mps')
elif torch.cuda.is_available():
    print("CUDA")
    device = torch.device('cuda')
else:
    print("WARNING CPU IN USE")
    device = torch.device('cpu')

dset = Arithmetic(max_val=1e4, test_data=True)
train_custom(dset, device, 4, 256, 32, 1024, 1e-3)


def test_search_space():
    """
    #FOR SAMIR TO RUN
    search_space(2, 10, device, dset)
    """
    #FOR LAPTOP
    search_space(4, 10, device, dset)
    """
    #FOR DESKTOP
    search_space(8, 10, device, dset)
    """

#test_search_space()
#print(top_n_params(4))
