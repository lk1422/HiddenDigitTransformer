from dataset import Arithmetic
from models import Base
from train import *
from hyperparam import search_space
import torch
import sys

if torch.backends.mps.is_available():
    device = torch.device('mps')
else if torch.cuda.is_available():
    device = torch.device('cuda')
else
    print("WARNING CPU IN USE")
    device = torch.device('cpu')

dset = Arithmetic(max_val=1e3, test_data=False)


def test_search_space():
    """
    FOR SAMIR TO RUN
    search_space(2, 10, device, dset)
    """
    """
    FOR LAPTOP
    #search_space(8, 10, device, dset)
    """
    """
    FOR DESKTOP
    search_space(16, 10, device, dset)
    """

test_search_space()
