import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as model_zoo
import numpy as np 
from resnet_simclr import get_resnet
from robust_resnet import get_robust_resnet50
from debiased_resnet import Model
    
def get_model_func(args, model):

    if args.dataset == 'camelyon17':
        n_classes = 2
    elif args.dataset == 'waterbird':
        n_classes = 2
    elif args.dataset == 'oh-65cls':
        n_classes = 65
    elif args.dataset == 'cifar-10':
        n_classes = 2
    else:
        raise NotImplementedError(f"Missing implementation for dataset {args.dataset}")
    

    if model == 'resnet50':
        def m_f():
            m = model_zoo.resnet50(pretrained=args.pretrained)
            d = m.fc.in_features
            m.fc = nn.Linear(d, n_classes)
            return m.to(args.device)
        return m_f
    elif model == 'resnet50_np':
        def m_f():
            m = model_zoo.resnet50(pretrained=False)
            d = m.fc.in_features
            m.fc = nn.Linear(d, n_classes)
            return m.to(args.device)
        return m_f         
    elif model == 'resnet18':
        def m_f():
            m = model_zoo.resnet18(pretrained=args.pretrained)
            d = m.fc.in_features
            m.fc = nn.Linear(d, n_classes)
            return m.to(args.device)
        return m_f
    elif model == "vit_b_16":
        def m_f():
            pretrained = None if not(args.pretrained) else 'IMAGENET1K_V1'
            m = model_zoo.vit_b_16(weights=pretrained)
            m.heads = nn.Linear(in_features=768, out_features=n_classes, bias=True)
            return m.to(args.device)
        
        return m_f
    elif model == "resnet50SwAV":
        def m_f():
            model = torch.hub.load('facebookresearch/swav:main', 'resnet50')
            d = model.fc.in_features
            model.fc = nn.Linear(d, n_classes)
            return model.to(args.device)
        return m_f
    elif model == "resnet50MocoV2":
        def m_f():
            state = torch.load("/datasets/home/hbenoit/mocov2/moco_v2_800ep_pretrain.pth.tar")
            new_state = {k.replace("module.encoder_q.",""):v for k,v in state["state_dict"].items()}
            for i in ["0","2"]:
                new_state.pop(f"fc.{i}.bias")
                new_state.pop(f"fc.{i}.weight")

            model = model_zoo.resnet50(pretrained=False)
            d = model.fc.in_features
            model.load_state_dict(new_state, strict=False)
            model.fc = nn.Linear(d, n_classes)
            return model.to(args.device)
        
        return m_f
    elif model == "resnet50SIMCLRv2":
        def m_f():
            model, _ = get_resnet(depth=50, width_multiplier=1, sk_ratio=0)
            state = torch.load("/datasets/home/hbenoit/SimCLRv2-Pytorch/r50_1x_sk0.pth")
            model.load_state_dict(state["resnet"])
            d = model.fc.in_features
            model.fc = nn.Linear(d, n_classes)
            return model.to(args.device)
        return m_f
    elif model == "resnet50Debiased":
        raise NotImplementedError
        def m_f():
            model = Model()
            state = torch.load("/datasets/home/hbenoit/debiased/256_model_400_pi01.pth")
            new_state = {k.replace("module.",""):v for k,v in state.items()}
            model.load_state_dict(state_dict=new_state, strict=False)
            return model.to(args.device)
        return m_f
    elif model == "robust_resnet50":
        def m_f():
            robust = get_robust_resnet50()
            state = torch.load("/datasets/home/hbenoit/robust_resnet/resnet50_l2_eps0.05.ckpt")
            new_state = {}
            for k in state["model"]:
                if "attacker" not in k:
                    new_state [k.replace("module.","")] = state["model"][k]
            robust.load_state_dict(new_state)
            d = robust.model.fc.in_features
            robust.model.fc = nn.Linear(d, n_classes)
            return robust.to(args.device)
        return m_f
    else:
        raise NotImplementedError(f"Missing implemntation for model '{model}'.")
    




