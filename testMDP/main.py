from .MDP import *
from PPO.base_model import *
import copy
from PPO.PPO import clippedLossVector
import numpy as np
import torch
import matplotlib.pyplot as plt
from .utils import *


device = torch.device('cuda')

def mainGridWorld():
    gw = GridWorld("testMDP/example_gw")
    model = BaseTokens(device, gw.states, gw.n_tokens, 4).to(device)
    x, tgt, rewards, acts = generate_sequence_grid_world(model, gw, 2, device)
    print(x)
    print(tgt)
    print("ACTIONS")
    print(acts)
    print(rewards)

if __name__ == "__main__":
    mainGridWorld()
