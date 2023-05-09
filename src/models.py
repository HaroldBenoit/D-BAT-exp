import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as model_zoo
import numpy as np 
from resnet_simclr import get_resnet
from robust_resnet import get_robust_resnet50
    
def get_model_func(args):
    if args.dataset == 'camelyon17':
        if args.model == 'resnet50':
            def m_f():
                m = model_zoo.resnet50(pretrained=args.pretrained)
                d = m.fc.in_features
                m.fc = nn.Linear(d, 2)
                return m.to(args.device)
            return m_f
        else:
            raise NotImplementedError(f"Missing implemntation for model '{args.model}'.")
    elif args.dataset == 'waterbird':
        if args.model == 'resnet50':
            def m_f():
                m = model_zoo.resnet50(pretrained=args.pretrained)
                d = m.fc.in_features
                m.fc = nn.Linear(d, 2)
                return m.to(args.device)
            return m_f
        elif args.model == 'resnet18':
            def m_f():
                m = model_zoo.resnet18(pretrained=args.pretrained)
                d = m.fc.in_features
                m.fc = nn.Linear(d, 2)
                return m.to(args.device)
            return m_f
        elif args.model == "resnet50SIMCLRv2":
            def m_f():
                model, _ = get_resnet(depth=50, width_multiplier=1, sk_ratio=0)
                state = torch.load("/datasets/home/hbenoit/SimCLRv2-Pytorch/r50_1x_sk0.pth")
                model.load_state_dict(state["resnet"])
                d = model.fc.in_features
                model.fc = nn.Linear(d, 2)
                return model.to(args.device)
            return m_f
        elif args.model == "robust_resnet50":
            def m_f():
                robust = get_robust_resnet50()
                state = torch.load("/datasets/home/hbenoit/robust_resnet/resnet50_l2_eps0.05.ckpt")
                new_state = {}
                for k in state["model"]:
                    if "attacker" not in k:
                        new_state [k.replace("module.","")] = state["model"][k]
                robust.load_state_dict(new_state)
                d = robust.model.fc.in_features
                robust.model.fc = nn.Linear(d, 2)
                return robust.to(args.device)
            return m_f
        else:
            raise NotImplementedError(f"Missing implemntation for model '{args.model}'.")
    elif args.dataset == 'oh-65cls':
        if args.model == 'resnet18':
            def m_f():
                m = model_zoo.resnet18(pretrained=args.pretrained)
                d = m.fc.in_features
                m.fc = nn.Linear(d, 65)
                return m.to(args.device)
            return m_f
        if args.model == 'resnet50':
            def m_f():
                m = model_zoo.resnet50(pretrained=args.pretrained)
                d = m.fc.in_features
                m.fc = nn.Linear(d, 65)
                return m.to(args.device)
            return m_f
        else:
            raise NotImplementedError(f"Missing implemntation for model '{args.model}'.")
    elif args.dataset == 'cifar-10':
        if args.model == 'resnet18':
            def m_f():
                m = model_zoo.resnet18(pretrained=args.pretrained)
                d = m.fc.in_features
                m.fc = nn.Linear(d, 2)
                return m.to(args.device)
            return m_f
        if args.model == 'resnet50':
            def m_f():
                m = model_zoo.resnet50(pretrained=args.pretrained)
                d = m.fc.in_features
                m.fc = nn.Linear(d, 2)
                return m.to(args.device)
            return m_f
        else:
            raise NotImplementedError(f"Missing implemntation for model '{args.model}'.")
    else:
        raise KeyError(f"Unknown dataseet '{args.dataset}'.")



