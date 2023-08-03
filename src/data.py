import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms
import torchvision
import torchvision.utils as vision_utils
from wilds import get_dataset as get_wild_dataset
from wilds.common.data_loaders import get_train_loader, get_eval_loader
from wilds.datasets.waterbirds_dataset import WaterbirdsDataset
from cifar_task import CIFARClassificationTask
import pandas as pd
from torch.utils.data import DataLoader, Dataset, random_split, Subset
import random

class WrappedDataLoader:
    def __init__(self, dl, func):
        self.dl = dl
        self.func = func
        self.dataset = self.dl.dataset

    def __len__(self):
        return len(self.dl)

    def __iter__(self):
        batches = iter(self.dl)
        for b in batches:
            yield (self.func(*b))

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def process_dataset_65_classes_office_home(datasets, device=None, shuffle=True):
    final_X, final_Y = [], []
    for dataset in datasets:
        for i, (x, y) in enumerate(dataset):
            final_X.append(x)
            final_Y.append(y)
    X = torch.stack(final_X)
    Y = torch.tensor(final_Y).long()
    if shuffle:
        perm = torch.randperm(len(X))
        X = X[perm]
        Y = Y[perm]
    if device is not None:
        X = X.to(device)
        Y = Y.to(device)
    return TensorDataset(X, Y)


def get_OH_65classes_v1(args):
    data_transform = transforms.Compose([
        transforms.Resize((90, 90)),
        transforms.ToTensor()
    ])

    data_train_1 = ImageFolder("./datasets/OfficeHomeDataset_10072016/Product", transform=data_transform)
    data_train_2 = ImageFolder("./datasets/OfficeHomeDataset_10072016/Clipart", transform=data_transform)
    data_train = process_dataset_65_classes_office_home([data_train_1, data_train_2], device=args.device, shuffle=False)

    data_test_1 = ImageFolder("./datasets/OfficeHomeDataset_10072016/Real World", transform=data_transform)
    data_test = process_dataset_65_classes_office_home([data_test_1], device=args.device, shuffle=False)
    data_test, data_valid, data_perturb = torch.utils.data.random_split(data_test, [1900, 500, 1957], 
                                                                        generator=torch.Generator().manual_seed(42))
    
    g = torch.Generator()
    g.manual_seed(args.seed)
    
    train_dl = DataLoader(data_train, batch_size=args.batch_size_train, worker_init_fn=seed_worker, generator=g, shuffle=True)
    valid_dl = DataLoader(data_valid, batch_size=args.batch_size_eval,worker_init_fn=seed_worker, generator=g, shuffle=False)
    test_dl = DataLoader(data_test, batch_size=args.batch_size_eval, worker_init_fn=seed_worker, generator=g,shuffle=False)
    perturb_dl = DataLoader(data_perturb, batch_size=args.batch_size_train, worker_init_fn=seed_worker, generator=g, shuffle=True)

    device_func = lambda x, y: {"x": x.to(args.device), "y": y.to(args.device)}
    train_dl = WrappedDataLoader(train_dl, device_func)
    valid_dl = WrappedDataLoader(valid_dl, device_func)
    test_dl = WrappedDataLoader(test_dl, device_func)
    perturb_dl = WrappedDataLoader(perturb_dl, device_func)
    
    return train_dl, valid_dl, test_dl, perturb_dl


def get_OH_65classes_v2(args):
    data_transform = transforms.Compose([
        transforms.Resize((90, 90)), # spatial size of vgg-f input
        transforms.ToTensor()
    ])

    data_train_1 = ImageFolder("./datasets/OfficeHomeDataset_10072016/Product", transform=data_transform)
    data_train_2 = ImageFolder("./datasets/OfficeHomeDataset_10072016/Clipart", transform=data_transform)
    data_train = process_dataset_65_classes_office_home([data_train_1, data_train_2], device=args.device, shuffle=True)

    data_test = ImageFolder("./datasets/OfficeHomeDataset_10072016/Real World", transform=data_transform)
    data_test = process_dataset_65_classes_office_home([data_test], device=args.device, shuffle=True)

    data_perturb = ImageFolder("./datasets/OfficeHomeDataset_10072016/Art", transform=data_transform)
    data_perturb = process_dataset_65_classes_office_home([data_perturb], device=args.device, shuffle=True)

    data_test, data_valid = torch.utils.data.random_split(data_test, [len(data_test)-800, 800], 
                                                          generator=torch.Generator().manual_seed(42))
    
    g = torch.Generator()
    g.manual_seed(args.seed)

    train_dl = DataLoader(data_train, batch_size=args.batch_size_train, worker_init_fn=seed_worker, generator=g, shuffle=True)
    valid_dl = DataLoader(data_valid, batch_size=args.batch_size_eval, worker_init_fn=seed_worker, generator=g, shuffle=False)
    test_dl = DataLoader(data_test, batch_size=args.batch_size_eval, worker_init_fn=seed_worker, generator=g, shuffle=False)
    perturb_dl = DataLoader(data_perturb, batch_size=args.batch_size_train, worker_init_fn=seed_worker, generator=g, shuffle=True)
    
    return train_dl, valid_dl, test_dl, perturb_dl

def majority_only_waterbirds_dataset(dataset:WaterbirdsDataset) -> WaterbirdsDataset:
    train_split_mask = dataset.split_array == dataset.split_dict["train"]
    train_split_idx = np.where(train_split_mask)[0]

    train_metadata_array = dataset.metadata_array[train_split_mask,:]

    #(train_metadata_array[:,0].numpy() == train_metadata_array[:,1].numpy()).mean()
    spurious = (train_metadata_array[:,0].numpy() == train_metadata_array[:,1].numpy())

    non_spurious_train_split_idx = train_split_idx[~spurious]
    ## little cheat to make the non_spurious train data points not attached to any split (train,val, test)
    dataset.split_array[non_spurious_train_split_idx] = -1

    return dataset

def inverse_correlated_unlabeled_waterbirds(dataset:WaterbirdsDataset) -> WaterbirdsDataset:
    val_split_mask = dataset.split_array == dataset.split_dict["val"]
    val_split_idx = np.where(val_split_mask)[0]

    val_metadata_array = dataset.metadata_array[val_split_mask,:]

    spurious = (val_metadata_array[:,0].numpy() == val_metadata_array[:,1].numpy())
    spurious_val_split_idx = val_split_idx[spurious]
    # little cheat to make the spurious val data points not attached to any split (train,val, test)
    dataset.split_array[spurious_val_split_idx] = -1
    
    return dataset

def get_waterbird_v1(args): # confounder_strength = 0.95
    scale = 256.0/224.0
    transform_list= [
        transforms.Resize((int(224*scale), int(224*scale))),
        transforms.CenterCrop((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]
    if args.grayscale:
        transform_list.append(transforms.Grayscale(num_output_channels=3))

    transform_test = transforms.Compose(transform_list)
    try:
        args.root_dir
        dataset = get_wild_dataset(dataset="waterbirds", download=False, root_dir=args.root_dir)
    except:
        dataset = get_wild_dataset(dataset="waterbirds", download=True, root_dir=args.root_dir)
    
    if args.majority_only:
        dataset = majority_only_waterbirds_dataset(dataset=dataset)


    if args.inverse_correlation:
        dataset = inverse_correlated_unlabeled_waterbirds(dataset=dataset)

    g = torch.Generator()
    g.manual_seed(args.seed)


    data_train = dataset.get_subset("train", transform=transform_test)
    data_test  = dataset.get_subset("test", transform=transform_test)
    data_valid = dataset.get_subset("val", transform=transform_test)
    data_perturb = data_valid
    
    train_dl = get_train_loader("standard", data_train, batch_size=args.batch_size_train, worker_init_fn=seed_worker, generator=g,  num_workers=8, pin_memory=True)
    perturb_dl = get_train_loader("standard", data_perturb, batch_size=args.batch_size_train, worker_init_fn=seed_worker, generator=g,  num_workers=8, pin_memory=True)
    test_dl = get_eval_loader("standard", data_test, batch_size=args.batch_size_eval, worker_init_fn=seed_worker, generator=g,  num_workers=5, pin_memory=True)
    valid_dl = get_eval_loader("standard", data_valid, batch_size=args.batch_size_eval, worker_init_fn=seed_worker, generator=g,  num_workers=5, pin_memory=True)

    device_func = lambda x, y, meta: {"x": x.to(args.device), "y": y.to(args.device), "meta": meta.to(args.device)}
    train_dl = WrappedDataLoader(train_dl, device_func)
    valid_dl = WrappedDataLoader(valid_dl, device_func)
    test_dl = WrappedDataLoader(test_dl, device_func)
    perturb_dl = WrappedDataLoader(perturb_dl, device_func)
    
    return train_dl, valid_dl, test_dl, perturb_dl


def get_camelyon17(args): 
    transform = transforms.Compose([
        transforms.ToTensor()
    ])  
    try:
        dataset = get_wild_dataset(dataset="camelyon17", download=False, root_dir=args.root_dir)
    except:
        dataset = get_wild_dataset(dataset="camelyon17", download=True, root_dir=args.root_dir)
    
    data_train = dataset.get_subset("train", transform=transform)
    data_test  = dataset.get_subset("test", transform=transform)
    data_valid = dataset.get_subset("val", transform=transform)
    try:
        data_perturb = get_wild_dataset(dataset="camelyon17", download=False, unlabeled=True, root_dir=args.root_dir)
    except:
        data_perturb = get_wild_dataset(dataset="camelyon17", download=True, unlabeled=True, root_dir=args.root_dir)

    if args.perturb_type == "ood_is_test":
        data_perturb = data_perturb.get_subset("test_unlabeled", transform=transform) # train_unlabeled, val_unlabeled, test_unlabeled
    elif args.perturb_type == "ood_is_not_test":
        data_perturb = data_perturb.get_subset("val_unlabeled", transform=transform) # train_unlabeled, val_unlabeled, test_unlabeled
    else:
        NotImplementedError(f"Version of perturbations '{args.perturb_type}' not implemented for dataset '{args.dataset}'.")

    g = torch.Generator()
    g.manual_seed(args.seed)
    
    train_dl = get_train_loader("standard", data_train, batch_size=args.batch_size_train, worker_init_fn=seed_worker, generator=g,  num_workers=8, pin_memory=True)
    perturb_dl = get_train_loader("standard", data_perturb, batch_size=args.batch_size_train, worker_init_fn=seed_worker, generator=g,  num_workers=8, pin_memory=True)
    test_dl = get_eval_loader("standard", data_test, batch_size=args.batch_size_eval, worker_init_fn=seed_worker, generator=g,  num_workers=5, pin_memory=True)
    valid_dl = get_eval_loader("standard", data_valid, batch_size=args.batch_size_eval, worker_init_fn=seed_worker, generator=g,  num_workers=5, pin_memory=True)
    
    device_func = lambda x, y, meta: {"x": x.to(args.device), "y": y.to(args.device), "meta": meta.to(args.device)}
    train_dl = WrappedDataLoader(train_dl, device_func)
    valid_dl = WrappedDataLoader(valid_dl, device_func)
    test_dl = WrappedDataLoader(test_dl, device_func)
    ## not the same because no y attribute
    perturb_dl = WrappedDataLoader(perturb_dl, lambda x, meta: {"x": x.to(args.device),"meta": meta.to(args.device)})
    
    return train_dl, valid_dl, test_dl, perturb_dl



def CIFAR_with_spurious(cls, args):
    """
    Modifies the given Dataset class to return a tuple data, target, index
    instead of just data, target.
    """

    spurious_labelling = pd.read_csv(args.discovered_tasks_path)[f"resnet18_d8-{args.split_spurious_task_idx}"].to_numpy()
    spurious_labelling = 1 - spurious_labelling

    def __getitem__(self, index):
        data, target = cls.__getitem__(self, index)
        return data, target, spurious_labelling[index]

    return type(cls.__name__, (cls,), {
        '__getitem__': __getitem__,
    })


def get_cifar_10(args):

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
        mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
        std=[x / 255.0 for x in [63.0, 62.1, 66.7]],
    )
    ])

    ## makes it so that both semantic labelling and spurious labelling are returned
    CIFAR10WithIndicies = CIFAR_with_spurious(torchvision.datasets.CIFAR10, args)

    try:
        dataset = CIFAR10WithIndicies(root="./datasets/", train=True, download=True, transform=transform)
        test_dataset = CIFAR10WithIndicies(root="./datasets/", train=False, download=True, transform=transform)

    except:
        dataset = CIFAR10WithIndicies(root="./datasets/", train=True, download=False, transform=transform)
        test_dataset = CIFAR10WithIndicies(root="./datasets/", train=False, download=False, transform=transform)



    ## labelling with semantic task
    semantic_task = CIFARClassificationTask(task_type="real", task_idx=args.split_semantic_task_idx)
    relabel = lambda target: semantic_task(target)
    dataset.targets = relabel(dataset.targets)
    test_dataset.targets = relabel(test_dataset.targets)

    ## adversarial-splitting

    group_info = torch.load(args.conditional_labelling_split).bool()
    splits = torch.load(args.train_val_split)
    train_split, val_split = splits
    labeled_train, labeled_val = [group_info[indices] for indices in splits]
    labeled_train_split, unlabeled_train_split = train_split[labeled_train], train_split[~labeled_train]

    splits = [labeled_train_split, unlabeled_train_split, val_split]

    labeled_dataset_train, unlabeled_dataset_train, dataset_val = [Subset(dataset,indices) for indices in splits]

    dataset_perturbed = unlabeled_dataset_train
    dataset_train = labeled_dataset_train

    g = torch.Generator()
    g.manual_seed(args.seed)

    train_dl = DataLoader(dataset_train, batch_size=args.batch_size_train, worker_init_fn=seed_worker, generator=g,  num_workers=8, shuffle=True)
    valid_dl = DataLoader(dataset_val, batch_size=args.batch_size_eval, worker_init_fn=seed_worker, generator=g,  num_workers=5, shuffle=False)
    test_dl = DataLoader(test_dataset, batch_size=args.batch_size_eval, worker_init_fn=seed_worker, generator=g,  num_workers=5, shuffle=False)
    perturb_dl = DataLoader(dataset_perturbed, batch_size=args.batch_size_train, worker_init_fn=seed_worker, generator=g,  num_workers=8, shuffle=True)

    device_func = lambda x, y, spurious_y: {"x" : x.to(args.device), "y": y.to(args.device), "spurious_y":spurious_y.to(args.device)}
    train_dl = WrappedDataLoader(train_dl, device_func)
    valid_dl = WrappedDataLoader(valid_dl, device_func)
    test_dl = WrappedDataLoader(test_dl,device_func )
    perturb_dl = WrappedDataLoader(perturb_dl, device_func)
    
    return train_dl, valid_dl, test_dl, perturb_dl

def get_dataset(args):
    if args.dataset == 'kaggle-bird':
        if args.perturb_type == 'ood_is_test':
            return get_kaggle_bird_dl_v1(args)
        else:
            NotImplementedError(f"Version of perturbations '{args.perturb_type}' not implemented for dataset '{args.dataset}'.")
    elif args.dataset == 'camelyon17':
        return get_camelyon17(args=args)
    elif args.dataset == 'waterbird':
        if args.perturb_type == 'ood_is_test':
            return get_waterbird_v1(args)
        else:
            NotImplementedError(f"Version of perturbations '{args.perturb_type}' not implemented for dataset '{args.dataset}'.")
    elif args.dataset == 'oh-65cls':
        if args.perturb_type == 'ood_is_test':
            return get_OH_65classes_v1(args)
        else:
            NotImplementedError(f"Version of perturbations '{args.perturb_type}' not implemented for dataset '{args.dataset}'.")
    elif args.dataset == "cifar-10":
        if args.perturb_type == 'ood_is_test':
            return get_cifar_10(args)
        else:
            NotImplementedError(f"Version of perturbations '{args.perturb_type}' not implemented for dataset '{args.dataset}'.")
    else:
        raise KeyError(f"Unknown dataset '{args.dataset}'.")
