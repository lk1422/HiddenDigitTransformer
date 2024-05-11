from .MDP import *
from PPO.base_model import *
import copy
from PPO.PPO import clippedLossSequential
import pickle
import numpy as np
import torch
import matplotlib.pyplot as plt
from .utils import *

device = torch.device('cuda')

def mainGridWorld():
    gw = GridWorld("testMDP/example_gw")
    model = BaseTokens(device, gw.n_tokens, gw.n_tokens, 4, dim=128).to(device)
    x, tgt, rewards, acts = generate_sequence_grid_world(model, gw, 2, device)
    l_fn = clippedLossSequential(0.3, 1, 0.15, gw.PAD, device)
    optim = torch.optim.Adam(model.parameters(), lr=3e-4, eps=1e-5)
    losses, rewards = trainPPO(model, device, gw, 1000, 32, 128, l_fn, optim)
    metrics = [losses, rewards]
    pickle.dump(metrics, open("metrics.pkl", "wb"))

def experiment():
    gw = GridWorld("testMDP/example_gw")
    model = BaseTokens(device, gw.n_tokens, gw.n_tokens, 4, dim=128)
    model.load_state_dict(torch.load("temp_model.pth"))
    model = model.to(device)
    right_of = np.arange(0,gw.states)[gw.destination.astype(np.bool_)][0] + 1
    src = torch.ones(1,1) * right_of
    dec = torch.ones(1,1) * right_of
    src = src.to(device).to(torch.int32)
    dec = dec.to(device).to(torch.int32)
    model.eval()
    p, v = model(src, dec, -1, value=True)
    print(p)
    print(v)


if __name__ == "__main__":
    mainGridWorld()
