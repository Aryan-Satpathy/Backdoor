# Defense Against Backdoor Attacks in SSL

## Introduction
This repository contains code for our submission to ICLR workshop 24. We implement backdoor-poisoning attacks and evaluate our defense against these attacks on Image Classification Task. Attacks implemented:
- [x] CTRL: [ICCV 2023](https://arxiv.org/abs/2210.07346)
- [x] FIBA: [CVPR 2022](https://arxiv.org/abs/2112.01148)
- [x] HTBA: [CVPR 2022](https://arxiv.org/abs/1910.00033)

Defenses implemented:
  - [x] Gaussian Blur (Invariance-Equivariance)
  - [x] Luminance (Exploiting evasiveness)

Datasets implemented:
- [x] CIFAR 10
- [x] CIFAR 100
- [ ] IMAGENET 100 (100 class subset of IMAGENET)

We show a successful and partly generalizable defense against backdoor attacks in SSL and lay the theoretical foundation for defense against backdoor attacks in Semantic Segmentation and Object Detection tasks.

## Installation and Requirements
- Download the repository from [anonymous4openscience](https://anonymous.4open.science/r/Backdoor-028B).
- Make a virtual environment (optional) (**recommended**)
    ```bash
    virtualenv <env_name>
    source <env_name>/bin/activate
    ```
- Install necessary libraries
    ```bash
    pip install -r requirements.txt
    ```

## Running The Code
We provide a bash script to run our program with appropriate command-line arguments. 
- Give permission
```bash
chmod +x run.sh
```
- Call `main_train.py`
```bash
bash run.sh <--args values>
```
Call `bash run.sh --help` if unsure about the arguments, available options or their meaning.

## Results

Results of all experiments are saved in a folder named `saves`. Each experiment will create a folder named `<job name>` set by `run.sh`. Each experiment folder contains model state-dicts and optimizer states saved every 100 epoch, and a `tfenvent` file containing tensorboard log. To view training progress and compare training curves:

```bash
tensorboard --logdir=saves
```
When repeating the same experiment with different hyperparameters, use `--suffix` option in `run.sh` to prevent overwriting log of previous experiment.

## To Do
- [ ] Implement ImageNet 100
- [ ] Implement MoCo v2
- [ ] Remove unnecessary command line arguments and refactor code.

#### NOTE: This repository is a fork of [CTRL's repository](https://github.com/meet-cjli/CTRL) with refactoring, implementation of defenses and other attacks.
