import torch
import numpy as np
import torch.nn.functional as F
from itertools import combinations
from collections import defaultdict
import matplotlib.pyplot as plt


def dl_to_sampler(dl):
    dl_iter = iter(dl)
    def sample():
        nonlocal dl_iter
        try:
            return next(dl_iter)
        except StopIteration:
            dl_iter = iter(dl)
            return next(dl_iter)
    return sample


@torch.no_grad()
def get_acc(model, dl):
    assert model.training == False
    acc = []
    for X, y in dl:
        acc.append(torch.argmax(model(X), dim=1) == y)
    acc = torch.cat(acc)
    acc = torch.sum(acc)/len(acc)
    return acc.item()


@torch.no_grad()
def get_acc_ensemble(ensemble, dl):
    assert all(model.training == False for model in ensemble)
    acc = []
    for X, y in dl:
        outs = [torch.softmax(model(X), dim=1) for model in ensemble]
        outs = torch.stack(outs, dim=0).mean(dim=0)
        acc.append(torch.argmax(outs, dim=1) == y)
    acc = torch.cat(acc)
    acc = torch.sum(acc)/len(acc)
    return acc.item()


def heatmap_fig(s, vmin=0.5, vmax=0.7):
    fig = plt.figure()
    plt.imshow(s, cmap='coolwarm', vmin=vmin, vmax=vmax)
    plt.colorbar()
    plt.axis('off')
    return fig


@torch.no_grad()
def get_ensemble_similarity(ensemble, dl):
    assert all(model.training == False for model in ensemble)

    num_models=len(ensemble)
    pairwise_indexes = list(combinations(range(num_models),2))
    sims = defaultdict(list)

    for X,y in dl:
        outs = []
        for model in ensemble:
            out = torch.softmax(model(X), dim=1) ## B*n_classes
            out = torch.argmax(out, dim=1)  ## B*1
            outs.append(out)
        
        for idx1, idx2 in pairwise_indexes:
            similarity = (outs[idx1] == outs[idx2]).float().cpu().mean()
            sims[(idx1,idx2)].append(similarity)

    sim_table = np.ones(shape=(num_models,num_models))
    for idx1, idx2 in pairwise_indexes:
        sim_table[idx1,idx2] = np.array(sims[(idx1,idx2)]).mean()
        ## for symmetry
        sim_table[idx2, idx1] = sim_table[idx1,idx2]

    return sim_table

    
