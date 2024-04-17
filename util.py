import torch

def generate_sequence(model, device, src, max_len, pad_idx, sos_idx, eos_idx):
    toks = [sos_idx]
    i = 0
    while toks[-1] != eos_idx and i < max_len-1:
        tgt = torch.tensor([toks], dtype=torch.int64).to(device)
        pred = model(src, tgt, pad_idx)
        next_ = torch.argmax(pred[0, -1], dim=-1).item()
        toks.append(next_)
        i+=1
    return toks

def answer(expr, device, dataset):
    tensor = dset.get_tensor(expr).to(device)
    output = generate_sequence(model, device, tensor, dset.get_max_len(), \
            dset.get_pad_idx(), dset.get_sos_idx(), dset.get_eos_idx())
    print(dset.get_str(output))


