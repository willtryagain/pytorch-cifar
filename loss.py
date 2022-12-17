import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

def cross_entropy(input, target):
    condition = torch.isclose(input, torch.zeros_like(input))
    linput = torch.where(condition, torch.tensor(0).float().cuda(), torch.log(input))
    if torch.isnan(linput).any():
        print("fknan")
    res = -torch.sum(target * linput, dim=1)
    if torch.isnan(res).any():
        print("res fknan")
    return -torch.sum(target * linput, dim=1)

class BanditLoss(nn.Module):
    def __init__(self, method='vanilla', alpha=0.5, eps=0.001, gamma=0.3, verbose=False) -> None:
        super().__init__()
        self.method = method
        self.alpha = alpha
        self.gamma = gamma
        self.eps = eps
        self.verbose = verbose

    def forward(self, inp, target, test=False):
        criterion =  nn.CrossEntropyLoss()

        if self.method == 'vanilla':
            return criterion(inp, target)
        elif self.method == 'uniform':
            C = inp.shape[1]
            correct_target_distrib = F.one_hot(target, num_classes=C)
            incorr_target_distrib = (torch.ones_like(correct_target_distrib) - correct_target_distrib)/(C - 1)
            
            
            flag = (torch.argmax(inp, dim=1) == target).bool()
            corr_loss, incorr_loss = torch.tensor(0), torch.tensor(0)
            if flag.any().item():
                corr_loss = criterion(inp[flag], correct_target_distrib.float()[flag])
            if (~flag).any().item():
                incorr_loss = criterion(inp[~flag], incorr_target_distrib.float()[~flag])
            return corr_loss + self.alpha * incorr_loss

        elif self.method == 'smooth':
            k = inp.shape[1]
            others = self.eps/(k - 1)
            pred_target_distrib = F.one_hot(torch.argmax(inp, dim=1), num_classes=k)

            P = (1 - self.gamma) * pred_target_distrib + self.gamma/k
            cumsum = np.cumsum(P.cpu().numpy(), axis=1)
            u = np.random.rand(len(cumsum), 1)
            y_ = torch.tensor((u < cumsum).argmax(axis = 1)).cuda()
            
            corr_pred_mask = (y_ == target).bool().repeat(k, 1).T
            corr_pos_mask = F.one_hot(y_, num_classes=k)
            ce_distrib = torch.full_like(inp, others)

            ce_distrib += (1 - self.eps - others)*torch.reciprocal(P) * \
                corr_pred_mask * corr_pos_mask
            assert (ce_distrib > 0).all()
           
            if self.verbose:
                print("inp", inp[0])
                print("corr", target[0])
                print()


            return criterion(inp, ce_distrib)
        else:
            raise AttributeError

