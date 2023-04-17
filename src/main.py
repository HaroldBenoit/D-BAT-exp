import os
import os.path as osp
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import torchvision
import torchvision.utils as vision_utils
import copy
import json
import math
import argparse
import random
import yaml
from tqdm import tqdm
import wandb

from models import get_model_func
from utils import get_acc_ensemble, get_acc, heatmap_fig, get_ensemble_similarity
from utils import dl_to_sampler
from data import get_dataset


def get_args():
    config_parser = argparse.ArgumentParser(description='Training Config', add_help=False)
    config_parser.add_argument('-c', '--config', default='', type=str, metavar='FILE', nargs="*",
                            help='YAML config file specifying default arguments')

    parser = argparse.ArgumentParser()
    # General training params
    parser.add_argument('--ensemble_size', default=2, type=int)
    parser.add_argument('--batch_size_train', default=256, type=int)
    parser.add_argument('--batch_size_eval', default=512, type=int)
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--device', default='cuda:0', type=str)
    parser.add_argument('--epochs', default=50, type=int)
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--l2_reg', default=0.0005, type=float)
    parser.add_argument('--scheduler', default='none', choices=['triangle', 'multistep', 'cosine', 'none'])
    parser.add_argument('--opt', default='adam', choices=['adamw', 'sgd'])
    parser.add_argument('--eval_freq', default=50, type=int) # in iterations
    parser.add_argument('--ckpt_freq', default=1, type=int) # in epochs
    parser.add_argument('--results_base_folder', default="./exps", type=str) # in epochs
    # Diversity params
    parser.add_argument('--no_diversity', action='store_true')
    parser.add_argument('--dbat_loss_type', default='v1', choices=['v1', 'v2'])
    parser.add_argument('--perturb_type', default='ood_is_test', choices=['ood_is_test', 'ood_is_not_test'])
    parser.add_argument('--alpha', default=1.0, type=float)    
    # Dataset and model
    parser.add_argument('--model', default='resnet50', choices=['resnet18', 'resnet50'])
    parser.add_argument('--dataset', default='camelyon17', choices=['waterbird', 'camelyon17', 'oh-65cls', 'cifar-10'])
    parser.add_argument('--split_semantic_task_idx', type=int, default=0,help= "") 
    parser.add_argument('--split_spurious_task_idx', type=int, default=0,help= "") 
    parser.add_argument('--conditional_labelling_split_folder',type=str, 
        help='Where the conditional labelling split is located Example: /datasets/home/liang/label-cond-as/adversarial-splits/scripts/label-cond-splits/v4/')
    
    parser.add_argument('--conditional_labelling_split_name_format', type=str, help="Format of the conditional_labelling_split name,\
                        Example: s_{split_semantic_task_idx}_d_{split_spurious_task_idx}_conditional_labelling_split_0.5.torch")
    
    parser.add_argument('--train_val_split_folder',type=str)
    parser.add_argument('--train_val_split_name_format',type=str)
    parser.add_argument('--discovered_tasks_path', type=str, help="path to the csv file containing the discovered tasks", default="../assets/tasks/discovered.csv")

    parser.add_argument('--pretrained', action="store_true", default=False)


    ## Logging params
    parser.add_argument('--group', type=str, default='D-BAT')
    parser.add_argument('--notes', type=str, default='')
    parser.add_argument('--tags', type=str, nargs='*', default=[])
    parser.add_argument('--project_name', type=str, default='Conditional-AS')
    parser.add_argument('--entity', type=str, default='task-discovery')
    parser.add_argument('--nologger', action='store_true', default=False)
    parser.add_argument('--resume_id', default="")
    parser.add_argument('--no_resume', dest='resume', action='store_false', default=True)

    args_config, remaining = config_parser.parse_known_args()
    if args_config.config:
        for config_file in args_config.config:
            with open(config_file, 'r') as f:
                cfg = yaml.safe_load(f)
                parser.set_defaults(**cfg)
    # The main arg parser parses the rest of the args, the usual
    # defaults will have been overridden if config file specified.
    args = parser.parse_args(remaining)

    if args.dataset == "cifar-10":
        conditional_labelling_split_name = args.conditional_labelling_split_name_format.format(
                split_semantic_task_idx=args.split_semantic_task_idx, split_spurious_task_idx=args.split_spurious_task_idx)
        conditional_labelling_split = osp.join(args.conditional_labelling_split_folder, conditional_labelling_split_name)


        train_val_split_name = args.train_val_split_name_format.format(
                split_semantic_task_idx=args.split_semantic_task_idx, split_spurious_task_idx=args.split_spurious_task_idx)
        train_val_split = osp.join(args.train_val_split_folder, train_val_split_name)

        args.__setattr__("conditional_labelling_split",conditional_labelling_split)
        args.__setattr__("train_val_split", train_val_split)
    
    return args

def eval_val_metrics(m:nn.Module, m_idx:int, valid_dl:DataLoader, args, itr:int, epoch:int, adv_loss, erm_loss, loss, ensemble_early_stopped, last_best_valid_acc, scheduler, stats):
    m.eval()
    valid_acc = get_acc(m, valid_dl)
    if not(args.nologger):
        wandb.log({f"valid/m{m_idx+1}_semantic_acc": valid_acc})


    p_s = f"[m{m_idx+1}] {epoch}:{itr} [train] erm-loss: {erm_loss.item():.3f},"  + \
          f" adv-loss: {adv_loss.item():.3f} [valid] acc: {valid_acc:.3f} "
    stats[f"m{m_idx+1}"]["valid-acc"].append((itr, valid_acc))
    stats[f"m{m_idx+1}"]["erm-loss"].append((itr, erm_loss.item()))
    stats[f"m{m_idx+1}"]["adv-loss"].append((itr, adv_loss.item()))
    if valid_acc > last_best_valid_acc:
        last_best_valid_acc = valid_acc
        ensemble_early_stopped[m_idx] = copy.deepcopy(m.state_dict())
    if itr != 0 and scheduler is not None:
        p_s += f"[lr] {scheduler.get_last_lr()[0]:.5f} "
    print(p_s)
    if math.isnan(loss.item()): 
        raise(ValueError("Loss is NaN. :("))
    m.train()

def train(get_model, get_opt, num_models, train_dl, valid_dl, test_dl, perturb_dl, get_scheduler=None, max_epoch=10, 
          eval_freq=400, ckpt_freq=1, ckpt_path="", alpha=1.0, use_diversity_reg=True, dbat_loss_type='v1', extra_args=None):
    
    ensemble = [get_model() for _ in range(num_models)]
    ensemble_early_stopped = [None for _ in range(num_models)]
    
    last_opt = None
    last_scheduler = None
    start_epoch = 0
    start_m_idx = 0
    last_best_valid_acc = -1
    itr = -1

    stats = {f"m{i+1}": {"valid-acc": [], "erm-loss": [], "adv-loss": []} for i in range(len(ensemble))}
    stats['ensemble-test-acc'] = None
    stats['ensemble-test-pgd-acc'] = None
    stats['ensemble-test-acc-es'] = None
    stats['ensemble-test-pgd-acc-es'] = None

    for model in ensemble:
        model.train()
        
    for m_idx in range(start_m_idx, num_models):
        m = ensemble[m_idx]
        
        for m_ in ensemble[:m_idx]:
            m_.eval()
        
        opt = get_opt(m.parameters())
        scheduler = get_scheduler(opt) if get_scheduler is not None else None
        
        perturb_sampler = dl_to_sampler(perturb_dl)
        
        for epoch in tqdm(range(start_epoch, max_epoch)):
            for x, y in train_dl:
                itr += 1
                
                x_tilde = perturb_sampler()[0]
                
                erm_loss = F.cross_entropy(m(x), y)
                
                if use_diversity_reg and m_idx != 0:
                    
                    if dbat_loss_type == 'v1':
                        adv_loss = []

                        p_1_s, indices = [], []
                        with torch.no_grad():
                            for m_ in ensemble[:m_idx]:
                                p_1 = torch.softmax(m_(x_tilde), dim=1)
                                p_1, idx = p_1.max(dim=1)
                                p_1_s.append(p_1)
                                indices.append(idx)

                        p_2 = torch.softmax(m(x_tilde), dim=1)
                        p_2_s = [p_2[torch.arange(len(p_2)), max_idx] for max_idx in indices]

                        for i in range(len(p_1_s)):
                            al = (- torch.log(p_1_s[i] * (1-p_2_s[i]) + p_2_s[i] * (1-p_1_s[i]) +  1e-7)).mean()
                            adv_loss.append(al)
                            
                    elif dbat_loss_type == 'v2':
                        adv_loss = []
                        p_2 = torch.softmax(m(x_tilde), dim=1)
                        p_2_1, max_idx = p_2.max(dim=1) # proba of class 1 for m
                        
                        with torch.no_grad():
                            p_1_s = [torch.softmax(m_(x_tilde), dim=1) for m_ in ensemble[:m_idx]]
                            p_1_1_s = [p_1[torch.arange(len(p_1)), max_idx] for p_1 in p_1_s] # probas of class 1 for m_
                            
                        for i in range(len(p_1_s)):
                            al = (- torch.log(p_1_1_s[i] * (1.0 - p_2_1) + p_2_1 * (1.0 - p_1_1_s[i]) +  1e-7)).mean()
                            adv_loss.append(al)
                        
                    else:
                        raise NotImplementedError(f"Unknown adversarial loss type: '{dbat_loss_type}'")
                else:
                    adv_loss = [torch.tensor([0]).to(x.device)]

                adv_loss = sum(adv_loss)/len(adv_loss)
                loss = erm_loss + alpha * adv_loss

                if not(args.nologger):
                    wandb.log({f"train/m{m_idx+1}_erm_loss":erm_loss.item(), f"train/m{m_idx+1}_adv_loss":adv_loss.item(), f"train/m{m_idx+1}_loss": loss.item()})

                opt.zero_grad()
                loss.backward()
                opt.step()
                if scheduler is not None:
                    scheduler.step()

                if itr % eval_freq == 0:
                    eval_val_metrics(m=m,m_idx=m_idx, valid_dl=valid_dl, args=args, itr=itr, epoch=epoch, adv_loss=adv_loss,
                                     erm_loss=erm_loss, loss=loss, ensemble_early_stopped=ensemble_early_stopped, last_best_valid_acc=last_best_valid_acc,scheduler=scheduler, stats=stats)
                
            if epoch % ckpt_freq == 0:
                torch.save({'ensemble': [model.state_dict() for model in ensemble], 
                            'ensemble_early_stopped': ensemble_early_stopped, 
                            'last_opt': opt.state_dict(),
                            'last_scheduler': scheduler.state_dict() if scheduler is not None else None,
                            'last_epoch': epoch,
                            'last_m_idx': m_idx,
                            'last_itr': itr,
                            'last_best_valid_acc': last_best_valid_acc,
                           }, ckpt_path)
                
            if epoch == (max_epoch -1):
                eval_val_metrics(m=m,m_idx=m_idx, valid_dl=valid_dl, args=args, itr=itr, epoch=epoch, adv_loss=adv_loss,
                                     erm_loss=erm_loss, loss=loss, ensemble_early_stopped=ensemble_early_stopped, last_best_valid_acc=last_best_valid_acc, scheduler=scheduler, stats=stats)
        
        itr = -1
        last_best_valid_acc = -1
        
    stats['test-acc'] = []
    for i, model in enumerate(ensemble): # test acc for each predictor in ensemble
        model.eval()  
        test_acc = get_acc(model, test_dl)
        if not(args.nologger):
            wandb.log({f"test/m{i+1}_semantic_acc": test_acc})     

        stats['test-acc'].append(test_acc)
        print(f"[test m{i+1}] test-acc: {test_acc:.3f}")
        
    test_acc_ensemble = get_acc_ensemble(ensemble, test_dl)
    if not(args.nologger):
        wandb.log({f"test/ensemble_semantic_acc": test_acc_ensemble})

    stats['ensemble-test-acc'] = test_acc_ensemble
    print(f"[test (last iterates ensemble)] test-acc: {test_acc_ensemble:.3f}") 
    
    test_acc_ensemble_per_ens_size = None
    if len(ensemble) > 2: # ensemble test accs for sub-ensembles
        test_acc_ensemble_per_ens_size = [get_acc_ensemble(ensemble[:ne], test_dl) for ne in range(2, len(ensemble)+1)]
        ens_gs = ", ".join([f"{x:.3f}" for x in test_acc_ensemble_per_ens_size])
        print(f"[test ensemble given size] {stats['test-acc'][0]:.3f}, {ens_gs}")
    stats['test_acc_ensemble_per_ens_size'] = test_acc_ensemble_per_ens_size


    ## pairwise similarities
    sim_table = get_ensemble_similarity(ensemble=ensemble, dl=test_dl)

    print(sim_table)

    if not(args.nologger):
        wandb.log({"test/similarity_heatmap":heatmap_fig(sim_table)})


    return stats


def main(args): 
    

    args.device = torch.device(args.device)
    
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    print(f"Loading dataset '{args.dataset}'")
    
    train_dl, valid_dl, test_dl, perturb_dl = get_dataset(args)
        
    print(f"Train dataset length: {len(train_dl.dataset)}")
    print(f"Valid dataset length: {len(valid_dl.dataset)}")
    print(f"Test dataset length: {len(test_dl.dataset)}")
    print(f"Perturbations dataset length: {len(perturb_dl.dataset)}")
    
    get_model = get_model_func(args)
    
    if args.opt == 'adamw':
        get_opt = lambda p: torch.optim.AdamW(p, lr=args.lr, weight_decay=0.05)
    else:
        get_opt = lambda p: torch.optim.SGD(p, lr=args.lr, momentum=0.9, weight_decay=args.l2_reg)#, nesterov=True)
    
    if args.scheduler != 'none':
        if args.scheduler == 'triangle':
            get_scheduler = lambda opt: torch.optim.lr_scheduler.CyclicLR(opt, 0, args.lr, 
                                                                          step_size_up=(len(train_dl)*args.epochs)//2, 
                                                                          mode='triangular', cycle_momentum=False)
        elif args.scheduler == 'cosine':
            get_scheduler = lambda opt: torch.optim.lr_scheduler.CyclicLR(opt, 0, args.lr, 
                                                                          step_size_up=(len(train_dl)*args.epochs)//2, 
                                                                          mode='cosine', cycle_momentum=False)
        elif args.scheduler == 'multistep':
            n_iters = len(train_dl)*args.epochs
            milestones = [0.25*n_iters, 0.5*n_iters, 0.75*n_iters] # hard-coded steps for now, suitable for resnet18
            get_scheduler = lambda opt: torch.optim.lr_scheduler.MultiStepLR(opt, milestones=milestones, gamma=0.3)
        else:
            raise NotImplementedError(f"Unknown scheduler type: {args.scheduler}.")
    else:
        get_scheduler = None

    
    exp_name = f"ep={args.epochs}_lrmax={args.lr}_alpha={args.alpha}_dataset={args.dataset}_perturb_type={args.perturb_type}" + \
               f"_model={args.model}_scheduler={args.scheduler}_seed={args.seed}_opt={args.opt}_ensemble_size={args.ensemble_size}" + \
               f"_no_diversity={args.no_diversity}_dbat_loss_type={args.dbat_loss_type}_weight_decay={args.l2_reg}_no_nesterov_"
    

    log_name= f"alpha={args.alpha}_dbat_loss_type={args.dbat_loss_type}_weight_decay={args.l2_reg}_opt={args.opt}" + \
               f"_ensemble_size={args.ensemble_size}_dataset={args.dataset}_model={args.model}_epochs={args.epochs}" 
    
    if args.dataset == "cifar-10":
        exp_name=f"{exp_name}_semantic_idx={args.split_semantic_task_idx}_spurious_idx={args.split_spurious_task_idx}"
        log_name=f"{log_name}_semantic_idx={args.split_semantic_task_idx}_spurious_idx={args.split_spurious_task_idx}"


    ckpt_path = f"{args.results_base_folder}/{args.dataset}/perturb={args.perturb_type}/{args.model}/ep{args.epochs}/{exp_name}"
    if not os.path.exists(ckpt_path):
        os.makedirs(ckpt_path)
    else:
        if os.path.isfile(f"{ckpt_path}/summary.json"): # the experiment was already completed
            sys.exit(0)

    
    ##logging 
    if not(args.nologger):
        os.makedirs(os.path.join(ckpt_path, "wandb"), exist_ok=True)
        wandb.login()
        wandb.init(name=log_name,
                   project=args.project_name,
                   entity=args.entity,
                   dir= ckpt_path,
                   tags= ["d-bat"] + args.tags,
                   group=args.group,
                   notes=args.notes,
                   id = args.resume_id if args.resume else None
                   )

        ## saving hyperparameters
        wandb.config = vars(args)


    print(f"\nTraining \n{vars(args)}\n")
    stats = train(get_model, get_opt, args.ensemble_size, train_dl, valid_dl, test_dl, perturb_dl, get_scheduler, max_epoch= args.epochs, 
                  eval_freq=args.eval_freq, ckpt_freq=1, ckpt_path=f"{ckpt_path}/ckpt.pt", alpha=args.alpha, 
                  use_diversity_reg=not args.no_diversity, dbat_loss_type=args.dbat_loss_type, extra_args=args)
    
    args.device = None
    stats['args'] = vars(args)
    
    with open(f"{ckpt_path}/summary.json", "w") as fs:
        json.dump(stats, fs)
        


if __name__ == "__main__":
    
    args = get_args()
    
    main(args)

    
    