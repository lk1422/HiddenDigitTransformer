
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class clippedLossSequential(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, model, model_old, eps, c_1, src, generated, rewards, pad_idx, eos_idx, device):

        model.eval()
        model_old.eval()

        N = generated.shape[0]
        S = generated.shape[1]

        #pad_mask = torch.logical_not(tgt == pad_idx).to(torch.int64).to(device)
        eos_mask  = (generated==eos_idx)
        first_eos = ((generated == eos_idx).cumsum(dim=1).cumsum(dim=1) == 1)
        eos_mask = torch.logical_xor(eos_mask.cumsum(dim=1), first_eos)
        eos_mask = torch.logical_not(eos_mask)

        policy, value = model(src, generated, pad_idx, value=True)
        policy_old    = model_old(src, generated, pad_idx)
        policy = F.softmax(policy, dim=1)
        policy_old = F.softmax(policy_old, dim=1)

        batch_index = torch.arange(N).unsqueeze(1).expand(N,S)
        state_index = torch.arange(S).expand(N,S)
        pi = policy[batch_index, state_index, generated]
        pi_old = policy_old[batch_index, state_index, generated]
        
        r = torch.exp(torch.log(pi) - torch.log(pi_old))
        r_clipped = torch.clamp(r, min=1-eps, max=1+eps)

        rewards = rewards.unsqueeze(1).expand(N,S)
        value = value.squeeze(-1)
        A = (rewards - value)

        #l_clip = (torch.min(r*A, r_clipped*A) * eos_mask)
        l_clip = torch.min(r*A, r_clipped*A) 
        l_clip = l_clip.sum()/(N*eos_mask.sum())

        rewards = rewards * eos_mask
        value = value * eos_mask
        rewards = rewards.reshape(-1,1)
        value   = value.reshape(-1,1)

        l_value = F.mse_loss(value, rewards)

        model.train()
        model_old.train()

        #print(l_value)
        #print(l_clip, l_value)
        #return l_value

        return -l_clip + c_1*l_value

class clippedLossSequential(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, model, model_old, eps, c_1, src, generated, rewards, pad_idx, eos_idx, device):

        model.eval()
        model_old.eval()

        N = generated.shape[0]
        S = generated.shape[1]

        #pad_mask = torch.logical_not(tgt == pad_idx).to(torch.int64).to(device)
        eos_mask  = (generated==eos_idx)
        first_eos = ((generated == eos_idx).cumsum(dim=1).cumsum(dim=1) == 1)
        eos_mask = torch.logical_xor(eos_mask.cumsum(dim=1), first_eos)
        eos_mask = torch.logical_not(eos_mask)

        policy, value = model(src, generated, pad_idx, value=True)
        policy_old    = model_old(src, generated, pad_idx)
        policy = F.softmax(policy, dim=1)
        policy_old = F.softmax(policy_old, dim=1)

        batch_index = torch.arange(N).unsqueeze(1).expand(N,S)
        state_index = torch.arange(S).expand(N,S)
        pi = policy[batch_index, state_index, generated]
        pi_old = policy_old[batch_index, state_index, generated]
        
        r = torch.exp(torch.log(pi) - torch.log(pi_old))
        r_clipped = torch.clamp(r, min=1-eps, max=1+eps)

        rewards = rewards.unsqueeze(1).expand(N,S)
        value = value.squeeze(-1)
        A = (rewards - value)

        #l_clip = (torch.min(r*A, r_clipped*A) * eos_mask)
        l_clip = torch.min(r*A, r_clipped*A) 
        l_clip = l_clip.sum()/(N*eos_mask.sum())

        rewards = rewards * eos_mask
        value = value * eos_mask
        rewards = rewards.reshape(-1,1)
        value   = value.reshape(-1,1)

        l_value = F.mse_loss(value, rewards)

        model.train()
        model_old.train()

        #print(l_value)
        #print(l_clip, l_value)
        #return l_value

        return -l_clip + c_1*l_value

