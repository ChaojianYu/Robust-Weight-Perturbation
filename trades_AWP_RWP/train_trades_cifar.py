from __future__ import print_function
import os
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable

import models
from utils import Logger, AverageMeter, accuracy
from utils_awp import TradesAWP
from utils_rwp import TradesRWP


parser = argparse.ArgumentParser(description='PyTorch CIFAR TRADES Adversarial Training')
parser.add_argument('--arch', type=str, default='WideResNet34')
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--test-batch-size', type=int, default=128, metavar='N',
                    help='input batch size for testing (default: 128)')
parser.add_argument('--epochs', type=int, default=200, metavar='N',
                    help='number of epochs to train')
parser.add_argument('--start_epoch', type=int, default=1, metavar='N',
                    help='retrain from which epoch')
parser.add_argument('--data', type=str, default='CIFAR10', choices=['CIFAR10', 'CIFAR100'])
parser.add_argument('--data-path', type=str, default='../cifar-data',
                    help='where is the dataset CIFAR-10')
parser.add_argument('--weight-decay', '--wd', default=5e-4,
                    type=float, metavar='W')
parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                    help='learning rate')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--norm', default='l_inf', type=str, choices=['l_inf', 'l_2'],
                    help='The threat model')
parser.add_argument('--epsilon', default=8, type=float,
                    help='perturbation')
parser.add_argument('--num-steps', default=10, type=int,
                    help='perturb number of steps')
parser.add_argument('--step-size', default=2, type=float,
                    help='perturb step size')
parser.add_argument('--beta', default=6.0, type=float,
                    help='regularization, i.e., 1/lambda in TRADES')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--model-dir', default='./model-cifar-wideResNet',
                    help='directory of model for saving checkpoint')
parser.add_argument('--resume-model', default='', type=str,
                    help='directory of model for retraining')
parser.add_argument('--resume-optim', default='', type=str,
                    help='directory of optimizer for retraining')
parser.add_argument('--save-freq', '-s', default=1000, type=int, metavar='N',
                    help='save frequency')

parser.add_argument('--awp-gamma', default=0.005, type=float,
                    help='whether or not to add parametric noise')
parser.add_argument('--awp-warmup', default=10, type=int,
                    help='We could apply AWP after some epochs for accelerating.')
parser.add_argument('--robustWP', default=1, type=int,
                    help='We could translate from awp to twp.')

args = parser.parse_args()
epsilon = args.epsilon / 255
step_size = args.step_size / 255
if args.awp_gamma <= 0.0:
    args.awp_warmup = np.infty
if args.data == 'CIFAR100':
    NUM_CLASSES = 100
else:
    NUM_CLASSES = 10

# settings
model_dir = args.model_dir
if not os.path.exists(model_dir):
    os.makedirs(model_dir)
use_cuda = not args.no_cuda and torch.cuda.is_available()
torch.manual_seed(args.seed)
device = torch.device("cuda" if use_cuda else "cpu")
kwargs = {'num_workers': 2, 'pin_memory': True} if use_cuda else {}

# setup data loader
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])
transform_test = transforms.Compose([
    transforms.ToTensor(),
])
trainset = getattr(datasets, args.data)(
    root=args.data_path, train=True, download=True, transform=transform_train)
testset = getattr(datasets, args.data)(
    root=args.data_path, train=False, download=True, transform=transform_test)
train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(testset, batch_size=args.test_batch_size, shuffle=False, **kwargs)


def perturb_input(model,
                  x_natural,
                  target,
                  step_size=0.003,
                  epsilon=0.031,
                  perturb_steps=10,
                  distance='l_inf',
                  PGD_attack=False):
    model.eval()
    batch_size = len(x_natural)
    if distance == 'l_inf':
        x_adv = x_natural.detach() + 0.001 * torch.randn(x_natural.shape).to(device).detach()
        for _ in range(perturb_steps):
            x_adv.requires_grad_()
            with torch.enable_grad():
                loss_kl = F.kl_div(F.log_softmax(model(x_adv), dim=1),
                                   F.softmax(model(x_natural), dim=1),
                                   reduction='sum')
            grad = torch.autograd.grad(loss_kl, [x_adv])[0]
            x_adv = x_adv.detach() + step_size * torch.sign(grad.detach())
            x_adv = torch.min(torch.max(x_adv, x_natural - epsilon), x_natural + epsilon)
            x_adv = torch.clamp(x_adv, 0.0, 1.0)
        if PGD_attack:
            x_adv_pgd = x_natural.detach() + 0.001 * torch.randn(x_natural.shape).to(device).detach()
            for _ in range(perturb_steps):
                x_adv_pgd.requires_grad_()
                with torch.enable_grad():
                    loss_pgd = F.cross_entropy(model(x_adv_pgd), target)
                grad = torch.autograd.grad(loss_pgd, [x_adv_pgd])[0]
                x_adv_pgd = x_adv_pgd.detach() + step_size * torch.sign(grad.detach())
                x_adv_pgd = torch.min(torch.max(x_adv_pgd, x_natural - epsilon), x_natural + epsilon)
                x_adv_pgd = torch.clamp(x_adv_pgd, 0.0, 1.0)
    elif distance == 'l_2':
        delta = 0.001 * torch.randn(x_natural.shape).to(device).detach()
        delta = Variable(delta.data, requires_grad=True)

        # Setup optimizers
        optimizer_delta = optim.SGD([delta], lr=epsilon / perturb_steps * 2)

        for _ in range(perturb_steps):
            adv = x_natural + delta

            # optimize
            optimizer_delta.zero_grad()
            with torch.enable_grad():
                loss = (-1) * F.kl_div(F.log_softmax(model(adv), dim=1),
                                       F.softmax(model(x_natural), dim=1),
                                       reduction='sum')
            loss.backward()
            # renorming gradient
            grad_norms = delta.grad.view(batch_size, -1).norm(p=2, dim=1)
            delta.grad.div_(grad_norms.view(-1, 1, 1, 1))
            # avoid nan or inf if gradient is 0
            # if (grad_norms == 0).any():
            #     delta.grad[grad_norms == 0] = torch.randn_like(delta.grad[grad_norms == 0])
            optimizer_delta.step()

            # projection
            delta.data.add_(x_natural)
            delta.data.clamp_(0, 1).sub_(x_natural)
            delta.data.renorm_(p=2, dim=0, maxnorm=epsilon)
        x_adv = Variable(x_natural + delta, requires_grad=False)
    else:
        x_adv = x_natural.detach() + 0.001 * torch.randn(x_natural.shape).to(device).detach()
        x_adv = torch.clamp(x_adv, 0.0, 1.0)
    if PGD_attack:
        return x_adv, x_adv_pgd
    else:
        return x_adv


def train(model, train_loader, optimizer, epoch, awp_adversary, rwp_adversary):

    print('epoch: {}'.format(epoch))

    for batch_idx, (data, target) in enumerate(train_loader):
        x_natural, target = data.to(device), target.to(device)

        # craft adversarial examples
        x_adv = perturb_input(model=model,
                              x_natural=x_natural,
                              target=target,
                              step_size=step_size,
                              epsilon=epsilon,
                              perturb_steps=args.num_steps,
                              distance=args.norm,
                              PGD_attack=False)

        model.train()
        # calculate adversarial weight perturbation
        if epoch >= args.awp_warmup:
            if args.robustWP:
                rwp = rwp_adversary.calc_awp(inputs_adv=x_adv, inputs_clean=x_natural, targets=target, beta=args.beta)
                rwp_adversary.perturb(rwp)
            else:
                awp = awp_adversary.calc_awp(inputs_adv=x_adv, inputs_clean=x_natural, targets=target, beta=args.beta)
                awp_adversary.perturb(awp)

        optimizer.zero_grad()
        logits_adv = model(x_adv)
        loss_robust = F.kl_div(F.log_softmax(logits_adv, dim=1),
                               F.softmax(model(x_natural), dim=1),
                               reduction='batchmean')
        # calculate natural loss and backprop
        logits = model(x_natural)
        loss_natural = F.cross_entropy(logits, target)
        loss = loss_natural + args.beta * loss_robust


        # update the parameters at last
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch >= args.awp_warmup:
            if args.robustWP:
                rwp_adversary.restore(rwp)
            else:
                awp_adversary.restore(awp)


def test(model, test_loader, criterion):
    #global best_acc
    model.eval()

    pgd_adv_losses = AverageMeter()
    pgd_adv_top1 = AverageMeter()
    adv_losses = AverageMeter()
    adv_top1 = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            inputs_adv, inputs_adv_pgd = perturb_input(model=model,
                                                       x_natural=inputs,
                                                       target=targets,
                                                       step_size=step_size,
                                                       epsilon=epsilon,
                                                       perturb_steps=args.num_steps*2,
                                                       distance=args.norm,
                                                       PGD_attack=True)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            outputs_adv = model(inputs_adv)
            loss_adv = criterion(outputs_adv, targets)
            outputs_adv_pgd = model(inputs_adv_pgd)
            loss_adv_pgd = criterion(outputs_adv_pgd, targets)

            prec1, prec5 = accuracy(outputs.data, targets.data, topk=(1, 5))
            losses.update(loss.item(), inputs.size(0))
            top1.update(prec1.item(), inputs.size(0))
            prec1_adv, prec5_adv = accuracy(outputs_adv.data, targets.data, topk=(1, 5))
            adv_losses.update(loss_adv.item(), inputs_adv.size(0))
            adv_top1.update(prec1_adv.item(), inputs_adv.size(0))
            prec1_adv_pgd, prec5_adv_pgd = accuracy(outputs_adv_pgd.data, targets.data, topk=(1, 5))
            pgd_adv_losses.update(loss_adv_pgd.item(), inputs_adv_pgd.size(0))
            pgd_adv_top1.update(prec1_adv_pgd.item(), inputs_adv_pgd.size(0))

    return losses.avg, top1.avg, adv_losses.avg, adv_top1.avg, pgd_adv_losses.avg, pgd_adv_top1.avg


def adjust_learning_rate(optimizer, epoch):
    """decrease the learning rate"""
    lr = args.lr
    if epoch >= 100:
        lr = args.lr * 0.1
    if epoch >= 150:
        lr = args.lr * 0.01
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def main():
    # init model, ResNet18() can be also used here for training
    model = nn.DataParallel(getattr(models, args.arch)(num_classes=NUM_CLASSES)).to(device)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    # We use a proxy model to calculate AWP, which does not affect the statistics of BN.
    proxy = nn.DataParallel(getattr(models, args.arch)(num_classes=NUM_CLASSES)).to(device)
    proxy_optim = optim.SGD(proxy.parameters(), lr=args.lr)

    proxy_proxy = nn.DataParallel(getattr(models, args.arch)(num_classes=NUM_CLASSES)).to(device)

    awp_adversary = TradesAWP(model=model, proxy=proxy, proxy_optim=proxy_optim, gamma=args.awp_gamma)
    rwp_adversary = TradesRWP(model=model, proxy_1=proxy_proxy, proxy_2=proxy, proxy_2_optim=proxy_optim, gamma=args.awp_gamma)

    criterion = nn.CrossEntropyLoss()

    logger = Logger(os.path.join(model_dir, 'log.txt'), title=args.arch)
    logger.set_names(['Learning Rate',
                      'PGD Adv Val Loss', 'Adv Val Loss', 'Nat Val Loss',
                      'PGD Adv Val Acc.', 'Adv Val Acc.', 'Nat Val Acc.'])

    if args.resume_model:
        model.load_state_dict(torch.load(args.resume_model, map_location=device))
    if args.resume_optim:
        optimizer.load_state_dict(torch.load(args.resume_optim, map_location=device))

    best_adv_val_acc = 0
    best_pgd_adv_val_acc = 0
    for epoch in range(args.start_epoch, args.epochs + 1):
        # adjust learning rate for SGD
        lr = adjust_learning_rate(optimizer, epoch)

        # adversarial training
        train(model, train_loader, optimizer, epoch, awp_adversary, rwp_adversary)

        # evaluation on natural examples
        print('================================================================')
        val_loss, val_acc, adv_val_loss, adv_val_acc, pgd_adv_val_loss, pgd_adv_val_acc = test(model, test_loader, criterion)
        print('================================================================')

        logger.append([lr, pgd_adv_val_loss, adv_val_loss, val_loss, pgd_adv_val_acc, adv_val_acc, val_acc])

        # save checkpoint
        if epoch % args.save_freq == 0 or epoch == args.epochs:
            torch.save(model.state_dict(),
                       os.path.join(model_dir, 'ours-model-epoch{}.pt'.format(epoch)))
            torch.save(optimizer.state_dict(),
                       os.path.join(model_dir, 'ours-opt-checkpoint_epoch{}.tar'.format(epoch)))
        # save best
        if adv_val_acc > best_adv_val_acc:
            torch.save(model.state_dict(), os.path.join(model_dir, 'model-best.pt'))
            best_adv_val_acc = adv_val_acc
        if pgd_adv_val_acc > best_pgd_adv_val_acc:
            torch.save(model.state_dict(), os.path.join(model_dir, 'model-best-pgd.pt'))
            best_pgd_adv_val_acc = pgd_adv_val_acc


if __name__ == '__main__':
    main()
