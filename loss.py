import torch
from torch import nn
import torch.nn.functional as F

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
    def __init__(self, method='vanilla', alpha=0.5) -> None:
        super().__init__()
        self.method = method
        self.alpha = alpha

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
            return self.alpha * corr_loss + (1 - self.alpha) * incorr_loss

        else:
            raise AttributeError
        # condition = input.sum(dim=1).int() != torch.ones(input.shape[0]).cuda()
        # assert torch.all(torch.isclose(input.sum(dim=1), torch.ones(input.shape[0]).cuda()))
        # my_ce = cross_entropy(input, correct_target_distrib).mean()
        # torch_ce = nn.CrossEntropyLoss()(inp, target)

        # if not torch.all(input >= 0):
        #     print("input prob")
        #     # print(inp)
        #     print("inp stats:", inp.min().item(), inp.max().item())
        #     print("input stats:", input.min().item(), input.max().item())


        # incorr_target_distrib = torch.zeros_like(correct_target_distrib)
        # print("input shape:", input.shape)
        # print("correct_target_distrib:", correct_target_distrib.shape)
        # print("incorr_target_distrib:", incorr_target_distrib.shape)

        # print("my_ce:", my_ce.item(), "torch_ce:", torch_ce.item())
        # if torch.isnan(my_ce).any():
            # print("stats:", input.max().item(), input.min().item())
        # print("diff", (my_ce - torch_ce).item())
        # return torch_ce
        return l.mean()
