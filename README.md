# Robust Weight Perturbation for Adversarial Training

This branch is developed for adversarial training with robust weight perturbation, the related paper is as follows:

    Robust Weight Perturbation for Adversarial Training[C]
    Chaojian Yu, Bo Han, Mingming Gong, Li Shen, Shiming Ge, Du Bo, Tongliang Liu
    IJCAI. 2022.

## Requisite

This code is implemented in PyTorch, and we have tested the code under the following environment settings:

- python = 3.7.3
- torch = 1.2.0
- torchvision = 0.4.0

## What is in this repository
Codes for our RWP-based adversarial training (AT-RWP) are in `AT_AWP_RWP`; RWP-based TRADES (TRADES-RWP) are in `trades_AWP_RWP`, and RWP-based RST (RST-RWP) are in `RST_AWP_RWP`:
- In `./AT_AWP_RWP`, the codes for CIFAR-10, CIFAR-100, and SVHN are in `train_cifar10.py`, `train_cifar100.py`, `train_svhn.py` respectively.
- In `./trades_AWP_RWP`, the codes for CIFAR-10 are in `train_trades_cifar.py`.
- In `./RST_AWP_RWP`, the codes for CIFAR-10 are in `robust_self_training.py`.

Robustness evaluation codes are in `Robustness Evaluation`:
- The codes for robustness evaluation on FGSM, PGD-20, PGD-100, C&W are in `eval_pgd.py`.
- The codes for robustness evaluation on AA are in `eval_aa.py`.

## How to use it

For AT-RWP with a PreAct ResNet-18 on CIFAR-10 under L_inf threat model (8/255), run codes as follows, 
```
python train_cifar10.py
``` 

For TRADES-RWP with a WRN-34-10 on CIFAR10 under L_inf threat model (8/255), run codes as follows,
```
python train_trades_cifar.py
```

For RST-RWP with a WRN-28-10 on CIFAR10 under L_inf threat model (8/255), you first need to download [500K unlabeled data from TinyImages (with pseudo-labels)](https://drive.google.com/open?id=1LTw3Sb5QoiCCN-6Y5PEKkq9C9W60w-Hi), and then run codes as follows,
```
python robust_self_training.py
```

## Citation
If you find our code helpful, please consider citing our work:

    @article{yu2022robust,
      title={Robust Weight Perturbation for Adversarial Training},
      author={Yu, Chaojian and Han, Bo and Gong, Mingming and Shen, Li and Ge, Shiming and Du, Bo and Liu, Tongliang},
      journal={arXiv preprint arXiv:2205.14826},
      year={2022}
    }

## Reference Code
[1] AT: https://github.com/locuslab/robust_overfitting

[2] TRADES: https://github.com/yaodongyu/TRADES/

[3] RST: https://github.com/yaircarmon/semisup-adv

[4] AWP: https://github.com/csdongxian/AWP

[5] AutoAttack: https://github.com/fra31/auto-attack
