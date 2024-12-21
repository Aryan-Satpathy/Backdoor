# Backdoor and Adversarial Robustness for SSL

<!--## Introduction -->
This repository contains the official code implementation for our paper [Towards Adversarial Robustness And Backdoor Mitigation in SSL](https://arxiv.org/abs/2403.15918). If you find this repo useful for your work, please [cite](https://github.com/Aryan-Satpathy/Backdoor/README.md#cite-our-paper) our paper.

We implement backdoor-poisoning attacks and evaluate our defense against these attacks on Image Classification Task. Backdoor attacks supported:
- [x] CTRL: [ICCV 2023](https://arxiv.org/abs/2210.07346)
- [x] FIBA: [CVPR 2022](https://arxiv.org/abs/2112.01148)
- [x] HTBA: [CVPR 2022](https://arxiv.org/abs/1910.00033)

Datasets supported:
- [x] CIFAR 10
- [x] CIFAR 100
- [ ] IMAGENET 100 (100 class subset of IMAGENET)

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
#### SSL methods to add
- [ ] [JEPA](https://ai.meta.com/blog/yann-lecun-ai-model-i-jepa/)
- [ ] MoCo v2

#### New Datasets / Benchmarks
- [ ] ImageNet
- [x] https://robustbench.github.io/

##### NOTE: This repository uses a lot of base code from [CTRL's repository](https://github.com/meet-cjli/CTRL). We refactor their code and implement other models, defenses and attacks. We also completely rewrite the pipeline using [lightly](https://github.com/lightly-ai/lightly) for cleaner and shorter code.

## License
This code has a GPL-style license.

## Cite our paper
```
@misc{satpathy2024adversarialrobustnessbackdoormitigation,
      title={Towards Adversarial Robustness And Backdoor Mitigation in SSL}, 
      author={Aryan Satpathy and Nilaksh Singh and Dhruva Rajwade and Somesh Kumar},
      year={2024},
      eprint={2403.15918},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2403.15918}, 
}
```
