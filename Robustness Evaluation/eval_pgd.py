import argparse
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms

import os

from wideresnet import WideResNet
from preactresnet import PreActResNet18


device = 'cuda' if torch.cuda.is_available() else 'cpu'


def replicate_input(x):
    return x.detach().clone()


def to_one_hot(y, num_classes=10):
    """
    Take a batch of label y with n dims and convert it to
    1-hot representation with n+1 dims.
    Link: https://discuss.pytorch.org/t/convert-int-into-one-hot-format/507/24
    """
    y = replicate_input(y).view(-1, 1)
    y_one_hot = y.new_zeros((y.size()[0], num_classes)).scatter_(1, y, 1)
    return y_one_hot


class CarliniWagnerLoss(nn.Module):
    """
    Carlini-Wagner Loss: objective function #6.
    Paper: https://arxiv.org/pdf/1608.04644.pdf
    """

    def __init__(self, kappa=10.0):
        super(CarliniWagnerLoss, self).__init__()
        self.kappa = kappa

    def forward(self, input, target):
        """
        :param input: pre-softmax/logits.
        :param target: true labels.
        :return: CW loss value.
        """
        num_classes = input.size(1)
        label_mask = to_one_hot(target, num_classes=num_classes).float()
        correct_logit = torch.sum(label_mask * input, dim=1)
        wrong_logit = torch.max((1. - label_mask) * input, dim=1)[0]
        loss = -F.relu(correct_logit - wrong_logit + self.kappa).sum()
        return loss


def clamp(X, lower_limit, upper_limit):
    return torch.max(torch.min(X, upper_limit), lower_limit)


def filter_state(state_dict):
    from collections import OrderedDict
    if 'state_dict' in state_dict.keys():
        state_dict = state_dict['state_dict']
    filtered_results = OrderedDict()
    for k, v in state_dict.items():
        if 'sub_block' in k:
            continue
        if 'module' in k:
            filtered_results[k[7:]] = v
        else:
            filtered_results[k] = v
    return filtered_results


def attack_pgd(model, X, y, epsilon, alpha, attack_iters, restarts,
               norm, early_stop=False, loss_type='CrossEntropyLoss',
               is_random=True):
    max_loss = torch.zeros(y.shape[0]).to(device)
    max_delta = torch.zeros_like(X).to(device)
    for _ in range(restarts):
        delta = torch.zeros_like(X).to(device)
        if is_random:
            if norm == "l_inf":
                delta.uniform_(-epsilon, epsilon)
            elif norm == "l_2":
                delta.normal_()
                d_flat = delta.view(delta.size(0),-1)
                n = d_flat.norm(p=2,dim=1).view(delta.size(0),1,1,1)
                r = torch.zeros_like(n).uniform_(0, 1)
                delta *= r/n*epsilon
            else:
                raise ValueError
        delta = clamp(delta, lower_limit-X, upper_limit-X)
        delta.requires_grad = True
        for _ in range(attack_iters):
            output = model(normalize(X + delta))
            if early_stop:
                index = torch.where(output.max(1)[1] == y)[0]
            else:
                index = slice(None,None,None)
            if not isinstance(index, slice) and len(index) == 0:
                break
            if loss_type == 'CrossEntropyLoss':
                loss = F.cross_entropy(output, y)
            elif loss_type == 'CWLoss':
                loss = CarliniWagnerLoss()(output, y)
            else:
                raise ValueError('Please use valid losses.')
            loss.backward()
            grad = delta.grad.detach()
            d = delta[index, :, :, :]
            g = grad[index, :, :, :]
            x = X[index, :, :, :]
            if norm == "l_inf":
                d = torch.clamp(d + alpha * torch.sign(g), min=-epsilon, max=epsilon)
            elif norm == "l_2":
                g_norm = torch.norm(g.view(g.shape[0],-1),dim=1).view(-1,1,1,1)
                scaled_g = g/(g_norm + 1e-10)
                d = (d + scaled_g*alpha).view(d.size(0),-1).renorm(p=2,dim=0,maxnorm=epsilon).view_as(d)
            d = clamp(d, lower_limit - x, upper_limit - x)
            delta.data[index, :, :, :] = d
            delta.grad.zero_()
        all_loss = F.cross_entropy(model(normalize(X+delta)), y, reduction='none')
        max_delta[all_loss >= max_loss] = delta.detach()[all_loss >= max_loss]
        max_loss = torch.max(max_loss, all_loss)
    return max_delta


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='xxx', type=str,
                        choices=['PreActResNet18', 'WideResNet28', 'WideResNet34', 'VGG19'])
    parser.add_argument('--batch-size', default=64, type=int)
    parser.add_argument('--dataset', default='cifar10', type=str, choices=['cifar10', 'cifar100', 'svhn'])
    parser.add_argument('--data-dir', default='../cifar-data', type=str)
    parser.add_argument('--attack', default='pgd', type=str, choices=['pgd', 'none'])
    parser.add_argument('--epsilon', default=8, type=int)
    parser.add_argument('--attack-iters', default=1, type=int)
    parser.add_argument('--restarts', default=1, type=int)
    parser.add_argument('--pgd-alpha', default=10, type=float)
    parser.add_argument('--norm', default='l_inf', type=str, choices=['l_inf', 'l_2'])
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--width-factor', default=10, type=int)
    parser.add_argument('--checkpoint', default='xxx/xxx.pth', type=str)
    parser.add_argument('--preprocess', default='meanstd', choices=['meanstd', '01', '+-1'])
    parser.add_argument('--no-random', action='store_false')
    parser.add_argument('--loss', default='CrossEntropyLoss', choices=['CrossEntropyLoss', 'CWLoss'])
    return parser.parse_args()


args = get_args()

if args.preprocess == 'meanstd':
    if args.dataset == 'cifar10':
        mu = (0.4914, 0.4822, 0.4465)
        std = (0.2471, 0.2435, 0.2616)
    elif args.dataset == 'cifar100':
        mu = (0.5070751592371323, 0.48654887331495095, 0.4409178433670343)
        std = (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)
elif args.preprocess == '01':
    mu = (0.0, 0.0, 0.0)
    std = (1.0, 1.0, 1.0)
elif args.preprocess == '+-1':
    mu = (0.5, 0.5, 0.5)
    std = (0.5, 0.5, 0.5)
else:
    raise ValueError('Please use valid parameters for normalization.')
mu = torch.tensor(mu).view(3,1,1).to(device)
std = torch.tensor(std).view(3,1,1).to(device)


def normalize(X):
    return (X - mu)/std


upper_limit, lower_limit = 1, 0


def main():

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    test_transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    num_workers = 2
    num_classes = int(args.dataset[5:])
    test_dataset = getattr(datasets, args.dataset.upper())(
        args.data_dir, train=False, transform=test_transform, download=True)
    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=2,
    )

    epsilon = (args.epsilon / 255.)
    pgd_alpha = (args.pgd_alpha / 255.)

    if args.model == 'PreActResNet18':
        model = PreActResNet18(num_classes=num_classes)
    elif args.model == 'WideResNet34':
        model = WideResNet(34, num_classes, widen_factor=args.width_factor, dropRate=0.0)
    elif args.model == 'WideResNet28':
        model = WideResNet(28, num_classes, widen_factor=args.width_factor, dropRate=0.0)
    else:
        raise ValueError("Unknown model")

    model = model.to(device)
    criterion = nn.CrossEntropyLoss()

    if args.checkpoint:
        state_dict = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(filter_state(state_dict))

    model.eval()
    test_loss = 0
    test_acc = 0
    test_robust_loss_fgsm = 0
    test_robust_acc_fgsm = 0
    test_robust_loss_pgd20 = 0
    test_robust_acc_pgd20 = 0
    test_robust_loss_pgd100 = 0
    test_robust_acc_pgd100 = 0
    test_robust_loss_cw = 0
    test_robust_acc_cw = 0
    test_n = 0

    start_time = time.time()
    for i, (X, y) in enumerate(test_loader):
        X = X.to(device)
        y = y.to(device)

        # Random initialization
        if args.attack == 'none':
            delta = torch.zeros_like(X)
        else:
            delta_fgsm = attack_pgd(model, X, y, epsilon, 16 / 255., 1, args.restarts, args.norm, early_stop=False,
                               loss_type='CrossEntropyLoss', is_random=args.no_random)
            delta_pgd20 = attack_pgd(model, X, y, epsilon, 2 / 255., 20, args.restarts, args.norm, early_stop=False,
                               loss_type='CrossEntropyLoss', is_random=args.no_random)
            delta_pgd100 = attack_pgd(model, X, y, epsilon, 2 / 255., 100, args.restarts, args.norm, early_stop=False,
                               loss_type='CrossEntropyLoss', is_random=args.no_random)
            delta_cw = attack_pgd(model, X, y, epsilon, 2 / 255., 100, args.restarts, args.norm, early_stop=False,
                               loss_type='CWLoss', is_random=args.no_random)
        delta_fgsm = delta_fgsm.detach()
        delta_pgd20 = delta_pgd20.detach()
        delta_pgd100 = delta_pgd100.detach()
        delta_cw = delta_cw.detach()

        robust_output_fgsm = model(normalize(torch.clamp(X + delta_fgsm[:X.size(0)], min=lower_limit, max=upper_limit)))
        robust_loss_fgsm = criterion(robust_output_fgsm, y)

        robust_output_pgd20 = model(normalize(torch.clamp(X + delta_pgd20[:X.size(0)], min=lower_limit, max=upper_limit)))
        robust_loss_pgd20 = criterion(robust_output_pgd20, y)

        robust_output_pgd100 = model(normalize(torch.clamp(X + delta_pgd100[:X.size(0)], min=lower_limit, max=upper_limit)))
        robust_loss_pgd100 = criterion(robust_output_pgd100, y)

        robust_output_cw = model(normalize(torch.clamp(X + delta_cw[:X.size(0)], min=lower_limit, max=upper_limit)))
        robust_loss_cw = criterion(robust_output_cw, y)

        output = model(normalize(X))
        loss = criterion(output, y)

        test_robust_loss_fgsm += robust_loss_fgsm.item() * y.size(0)
        test_robust_acc_fgsm += (robust_output_fgsm.max(1)[1] == y).sum().item()
        test_robust_loss_pgd20 += robust_loss_pgd20.item() * y.size(0)
        test_robust_acc_pgd20 += (robust_output_pgd20.max(1)[1] == y).sum().item()
        test_robust_loss_pgd100 += robust_loss_pgd100.item() * y.size(0)
        test_robust_acc_pgd100 += (robust_output_pgd100.max(1)[1] == y).sum().item()
        test_robust_loss_cw += robust_loss_cw.item() * y.size(0)
        test_robust_acc_cw += (robust_output_cw.max(1)[1] == y).sum().item()
        test_loss += loss.item() * y.size(0)
        test_acc += (output.max(1)[1] == y).sum().item()
        test_n += y.size(0)
    test_time = time.time() - start_time
    print('{}\t{:4f}\t{:4f}\t{:4f}\t{:4f}\t{:4f}\t{:4f}\t{:4f}\t{:4f}\t{:4f}\t{:4f}'.format(
        test_time, test_robust_loss_fgsm/test_n, test_robust_acc_fgsm/test_n,
                   test_robust_loss_pgd20/test_n, test_robust_acc_pgd20/test_n,
                   test_robust_loss_pgd100/test_n, test_robust_acc_pgd100/test_n,
                   test_robust_loss_cw/test_n, test_robust_acc_cw/test_n,
                   test_loss/test_n, test_acc/test_n))


if __name__ == "__main__":
    main()
