import torch
from collections import OrderedDict
import torch.nn as nn
import torch.nn.functional as F
EPS = 1E-20


def diff_in_weights(proxy_1, proxy_2):
    diff_dict = OrderedDict()
    proxy_1_state_dict = proxy_1.state_dict()
    proxy_2_state_dict = proxy_2.state_dict()
    for (old_k, old_w), (new_k, new_w) in zip(proxy_1_state_dict.items(), proxy_2_state_dict.items()):
        if len(old_w.size()) <= 1:
            continue
        if 'weight' in old_k:
            diff_w = new_w - old_w
            #diff_dict[old_k] = old_w.norm() / (diff_w.norm() + EPS) * diff_w
            diff_dict[old_k] = diff_w
    return diff_dict


def add_into_diff(model, diff_step, diff):
    diff_scale = OrderedDict()
    if not diff:
        diff = diff_step
        names_in_diff = diff_step.keys()
        diff_squeue = []
        w_squeue = []
        for name, param in model.named_parameters():
            if name in names_in_diff:
                diff_squeue.append(diff[name].view(-1))
                w_squeue.append(param.view(-1))
        diff_squeue_all = torch.cat(diff_squeue)
        w_squeue_all = torch.cat(w_squeue)
        scale_value = w_squeue_all.norm() / (diff_squeue_all.norm() + EPS)
        for name, param in model.named_parameters():
            if name in names_in_diff:
                diff_scale[name] = scale_value * diff[name]
    else:
        names_in_diff = diff_step.keys()
        for name in names_in_diff:
            diff[name] = diff[name] + diff_step[name]
        diff_squeue = []
        w_squeue = []
        for name, param in model.named_parameters():
            if name in names_in_diff:
                diff_squeue.append(diff[name].view(-1))
                w_squeue.append(param.view(-1))
        diff_squeue_all = torch.cat(diff_squeue)
        w_squeue_all = torch.cat(w_squeue)
        scale_value = w_squeue_all.norm() / (diff_squeue_all.norm() + EPS)
        for name, param in model.named_parameters():
            if name in names_in_diff:
                diff_scale[name] = scale_value * diff[name]
    return diff, diff_scale


def add_into_weights(model, diff, coeff=1.0):
    names_in_diff = diff.keys()
    with torch.no_grad():
        for name, param in model.named_parameters():
            if name in names_in_diff:
                param.add_(coeff * diff[name])


class RobustWeightPerturb(object):
    def __init__(self, model, proxy_1, proxy_2, proxy_2_optim, gamma):
        super(RobustWeightPerturb, self).__init__()
        self.model = model
        self.proxy_1 = proxy_1
        self.proxy_2 = proxy_2
        self.proxy_2_optim = proxy_2_optim
        self.gamma = gamma

    def calc_awp(self, inputs_adv, targets, LSC_loss):
        diff = OrderedDict()
        diff_scale = OrderedDict()

        for ii in range(10):
            self.proxy_1.load_state_dict(self.model.state_dict())
            self.proxy_2.load_state_dict(self.model.state_dict())
            add_into_weights(self.proxy_1, diff_scale, coeff=1.0 * self.gamma)
            add_into_weights(self.proxy_2, diff_scale, coeff=1.0 * self.gamma)
            self.proxy_2.train()
            #loss = - F.cross_entropy(self.proxy_2(inputs_adv), targets)
            output = self.proxy_2(inputs_adv)
            loss = nn.CrossEntropyLoss(reduce=False)(output, targets)
            Indicator = (loss < LSC_loss).cuda().type(torch.cuda.FloatTensor)
            loss = loss.mul(Indicator).mean()
            loss = -1 * loss
            self.proxy_2_optim.zero_grad()
            loss.backward()
            self.proxy_2_optim.step()

            diff_step = diff_in_weights(self.proxy_1, self.proxy_2)
            diff, diff_scale = add_into_diff(self.model, diff_step, diff)
        return diff_scale

    def perturb(self, diff):
        add_into_weights(self.model, diff, coeff=1.0 * self.gamma)

    def restore(self, diff):
        add_into_weights(self.model, diff, coeff=-1.0 * self.gamma)




