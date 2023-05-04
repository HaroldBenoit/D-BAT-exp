import torch
import numpy as np
import torch.nn.functional as F
from itertools import combinations
from collections import defaultdict
import matplotlib.pyplot as plt

def collate_list(vec):
    """
    If vec is a list of Tensors, it concatenates them all along the first dimension.

    If vec is a list of lists, it joins these lists together, but does not attempt to
    recursively collate. This allows each element of the list to be, e.g., its own dict.

    If vec is a list of dicts (with the same keys in each dict), it returns a single dict
    with the same keys. For each key, it recursively collates all entries in the list.
    """
    if not isinstance(vec, list):
        raise TypeError("collate_list must take in a list")
    elem = vec[0]
    if torch.is_tensor(elem):
        return torch.cat(vec)
    elif isinstance(elem, list):
        return [obj for sublist in vec for obj in sublist]
    elif isinstance(elem, dict):
        return {k: collate_list([d[k] for d in vec]) for k in elem}
    else:
        raise TypeError("Elements of the list to collate must be tensors or dicts.")

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
def get_acc(model, dl, return_logits=False):
    assert model.training == False
    acc = []
    spurious_acc = []
    logits_list=[]
    for batch in dl:
        X = batch["x"]
        y = batch["y"]
        logits= model(X)
        if return_logits:
            logits_list.append(logits.tolist())
        preds = torch.argmax(logits, dim=1)
        acc.append( preds == y)
        if "spurious_y" in batch:
            spurious_acc.append(preds == batch["spurious_y"])
    acc = torch.cat(acc)
    acc = torch.sum(acc)/len(acc)

    res = {"acc": acc.item()}

    if len(spurious_acc) != 0:
        spurious_acc = torch.cat(spurious_acc).flatten()
        spurious_acc = torch.sum(spurious_acc)/len(spurious_acc)
        res["spurious_acc"] = spurious_acc.item()

    if return_logits:
        res["logits_list"] = logits_list

    return res


@torch.no_grad()
def get_acc_ensemble(ensemble, dl, return_meta=False):
    assert all(model.training == False for model in ensemble)
    acc = []
    metas=[]
    for batch in dl:
        X = batch["x"]
        y = batch["y"]
        if return_meta and "meta" in batch:
            metas.append(batch["meta"].tolist())
        outs = [torch.softmax(model(X), dim=1) for model in ensemble]
        outs = torch.stack(outs, dim=0).mean(dim=0)
        acc.append(torch.argmax(outs, dim=1) == y)
    acc = torch.cat(acc)
    acc = torch.sum(acc)/len(acc)
    res = {"acc": acc.item()}
    if return_meta:
        res["metas"] = metas
    return res


def heatmap_fig(s, vmin=0.0, vmax=1.0):
    fig = plt.figure()
    plt.imshow(s, cmap='coolwarm', vmin=vmin, vmax=vmax)
    plt.colorbar()
    plt.axis('off')
    return fig


@torch.no_grad()
def get_batchwise_ensemble_similarity_logs(ensemble, x_tilde):
    preds = []

    for model in ensemble:
        model.eval()
        out = torch.softmax(model(x_tilde), dim=1) ## B*n_classes
        pred = torch.argmax(out, dim=1)  ## B*1
        preds.append(pred)

    logs = {}

    sims= []
    pairwise_indexes = list(combinations(range(len(ensemble)),2))
    for idx1, idx2 in pairwise_indexes:
        similarity = (preds[idx1] == preds[idx2]).float().cpu().mean()
        sims.append(similarity)

    sims = np.array(sims)
    logs[f"unlabeled/similarity_mean_{len(ensemble)}"] = sims.mean()
    logs[f"unlabeled/similarity_min_{len(ensemble)}"] = sims.min()
    logs[f"unlabeled/similarity_max_{len(ensemble)}"] = sims.max()

    # for model in ensemble:
    #     model.train()

    return logs

@torch.no_grad()
def get_ensemble_similarity(ensemble, dl, num_trained_models):
    for model in ensemble:
        model.eval()
    assert all(model.training == False for model in ensemble)

    num_models=len(ensemble)
    pairwise_indexes = list(combinations(range(num_trained_models),2))
    sims = defaultdict(list)

    for batch in dl:
        X = batch["x"]
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

    for model in ensemble:
        model.train()

    return sim_table

    
