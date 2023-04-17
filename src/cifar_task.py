import torch
import torch.nn.functional as F
import numpy as np

from abc import abstractmethod
from typing import List, OrderedDict
from torch import nn
from itertools import combinations
from abc import ABC
from torch import Tensor


CIFAR_REAL_BIN_TASKS = [cls1 for cls1 in list(combinations(range(10), 5))[:126]]
# TINY_IMAGENET_REAL_BIN_TASKS = np.load('assets/tasks/tiny_imagenet_binary_tasks.npy').tolist() + [list(range(100)) for _ in range(130)]
CORE_REAL_TASKS_IDX = [0, 11, 18, 27, 31, 38, 40]


class Task(ABC, nn.Module):
    DIM = None

    def __init__(self):
        super().__init__()

    @abstractmethod    
    def loss(self, a, b):
        pass

    @abstractmethod
    def metrics(self, a, b):
        pass

class BaseClassificationTask(Task):
    def loss(self, prediction, target):
        assert (prediction.dim() == 2) & (target.shape[0] == prediction.shape[0]), f'{prediction.shape=}, {target.shape=}'
        if target.dim() == 1:
            # assume targets are classes
            return F.cross_entropy(prediction, target)
        elif target.dim() == 2:
            # assume targets are probabilities 
            return self.cross_entropy_loss(target, prediction)

    def metrics(self, prediction, target):
        labels = target if target.dim() == 1 else target.argmax(1)
        # TODO: change to bar plot/table
        p_classes = prediction.argmax(1)
        rates = {
            f'rate_{c}': (p_classes == c).float().mean().detach() for c in range(prediction.shape[1])
        }

        return {
            'cross_entropy': self.loss(prediction, target).item(),
            'acc': (prediction.argmax(1) == labels).float().mean().item(),
            **rates,
        }

    @staticmethod
    def cross_entropy_loss(target, q):
        assert target.dim() == 2 and q.dim() == 2
        loss = -(F.softmax(target, 1) * F.log_softmax(q, 1)).sum(1)
        # loss = (F.softmax(p, 1) * F.log_softmax(q, 1)).sum(1) + (F.softmax(p, 1) * F.log_softmax(q, 1)).sum(1)
        return loss.mean()
    


class CIFARClassificationTask(BaseClassificationTask):
    N = 50000
    DIM = 2

    def __init__(
        self,
        task_type: str ='random',
        task_idx: int = 0,
        net_arch: str = 'resnet18',
        dataset: str = 'cifar10',
        n_classes: int = 2,
    ):
        super().__init__()

        self.task_type = task_type            
        self.DIM = n_classes

        assert self.DIM in [2], f'{self.DIM}-way classification tasks are NOT supported'

        label2val = [int(i in CIFAR_REAL_BIN_TASKS[task_idx]) for i in range(10)]
        table = torch.LongTensor(label2val)
        print(f'[TASK] ===> Real task: {table[:20]}')
        self.lookup_table = nn.parameter.Parameter(table, requires_grad=False)

    def forward(self, y):
        assert y is not None
        t = self.lookup_table[y]
        return t

    def load_state_dict(self, state_dict: OrderedDict[str, Tensor], strict: bool = True):
        res = super().load_state_dict(state_dict, strict=strict)

        print(f'[TASK] ===> Loaded "{self.task_type}" task: example={self.lookup_table[:20].tolist()}, mean={self.lookup_table.float().mean():.2f}')
        return res
