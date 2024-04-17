import torch
import torch.nn as nn
import torch.nn.functional as F

def search_hyperparameters():
    pass
def get_accuracy(model, dset, batch_size, test=False):
    ##Fix to ignore Pad accuracy
    pad_idx = dset.get_pad_idx()
    x,y = get_batch(batch_size, test=test)
    y_ = torch.concat((y[:, 1:], torch.ones(y.shape[0], 1).to(device) * pad_idx), dim=1)
    out = model(x,y, pad_idx)
    return (out == y_).sum() / (torch.ones_like(out).sum())

def eval(model, dset):

def train(model, device, epoch, batch_size, n_iterations, loss, optim, dset, metrics):
    pad_idx = dset.get_pad_idx()
    for e in range(epoch):
        avg_loss = 0
        for n in range(n_iterations):
            x, y = dset.get_batch(batch_size)
            x, y = x.to(device), y.to(device)
            y_ = torch.concat((y[:, 1:], torch.ones(y.shape[0], 1).to(device) * pad_idx), dim=1)
            optim.zero_grad()
            out = model(x, y, pad_idx)
            y_ = y_.reshape(-1).to(torch.int64)
            out = out.reshape(-1, out.shape[-1])
            l = loss(out, y_)
            avg_loss+=l.item()
            l.backward()
            optim.step()
        print(avg_loss/n_iterations)



