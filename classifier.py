import os
import sys 
import argparse
import warnings 

from utils.frequency import PoisonFre

import torch.optim as optim
import torch.backends.cudnn as cudnn 
import torchvision
from torch.utils.data import DataLoader
from torchvision import transforms
from torch import nn


sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from methods import set_model
from methods.base import CLTrainer
from utils.util import *
from loaders.diffaugment import set_aug_diff, PoisonAgent

parser = argparse.ArgumentParser(description='CTRL Training')


### dataloader
parser.add_argument('--data_path', default='data/')
parser.add_argument('--dataset', default='cifar10', choices=['cifar10', 'cifar100', 'imagenet100'])
parser.add_argument('--image_size', default = 32, type=int)
parser.add_argument('--disable_normalize', action='store_true', default=True)
parser.add_argument('--full_dataset', action='store_true', default=True)
parser.add_argument('--window_size', default = 32, type=int)
parser.add_argument('--eval_batch_size', default = 512, type=int)
parser.add_argument('--num_workers', default=4, type=int)


### training
parser.add_argument('--arch', default='resnet18', type=str, choices=['resnet18', 'resnet50', 'resnet101', 'shufflenet', 'mobilenet', 'squeezenet'])
parser.add_argument('--method', default = 'simclr', choices=['simclr',  'byol', 'moco', 'simsiam'])
parser.add_argument('--batch_size', default = 1024, type=int)
parser.add_argument('--epochs', default = 1000, type=int)
parser.add_argument('--start_epoch', default = 0, type=int)
parser.add_argument('--remove', default = 'none', choices=['crop', 'flip', 'color', 'gray', 'none'])
parser.add_argument('--poisoning', action='store_true', default=False)
parser.add_argument('--update_model', action='store_true', default=False)
parser.add_argument('--contrastive', action='store_true', default=False)
parser.add_argument('--knn_eval_freq', default=1, type=int)
parser.add_argument('--distill_freq', default=5, type=int)
parser.add_argument('--saved_path', default='none', type=str)
parser.add_argument('--mode', default='normal', choices=['normal', 'frequency'])


## ssl setting
parser.add_argument('--temp', default=0.5, type=float)
parser.add_argument('--lr', default=0.06, type=float)
parser.add_argument('--wd', default=5e-4, type=float)
parser.add_argument('--cos', action='store_true', default=True)
parser.add_argument('--byol-m', default=0.996, type=float)
parser.add_argument('--moco-m', default=0.999, type=float)


###poisoning
parser.add_argument('--poisonkey', default=7777, type=int)
parser.add_argument('--target_class', default=1, type=int)
parser.add_argument('--poison_ratio', default = 0.01, type=float)
parser.add_argument('--pin_memory', action='store_true', default=False)
parser.add_argument('--select', action='store_true', default=False)
parser.add_argument('--reverse', action='store_true', default=False)
parser.add_argument('--trigger_position', nargs ='+', type=int)
parser.add_argument('--magnitude', default = 100.0, type=float)
parser.add_argument('--trigger_size', default=5, type=int)
parser.add_argument('--channel', nargs ='+', type=int)
parser.add_argument('--threat_model', default='our', choices=['ctrl', 'htba', 'fiba'])
parser.add_argument('--loss_alpha', default = 2.0, type=float)
parser.add_argument('--strength', default= 1.0, type=float)  ### augmentation strength

# Patch
parser.add_argument('--trigger_width', default = 8, type=int)

###logging
parser.add_argument('--log_path', default='Experiments', type=str, help='path to save log')
parser.add_argument('--poison_knn_eval_freq', default=5, type=int)
parser.add_argument('--poison_knn_eval_freq_iter', default=1, type=int)
parser.add_argument('--debug', action='store_true', default=False)
parser.add_argument('--trial', default='0', type=str)

###others
parser.add_argument('--distributed', action='store_true',
                    help='distributed training')
parser.add_argument('--gpu', default= 0, type=int,
                    help='GPU id to use.')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')

###our arguments
parser.add_argument('--blur', action='store_true',
                    help='blur augmentation', default=False)
parser.add_argument('--freqPatch', action='store_true',
                    help='freqPatch augmentation', default=False)
parser.add_argument('--value_channel', action='store_true',
                    help='value augmentation', default=False)
### defense
parser.add_argument(
    "--ctype",
    default="RGB",
    choices=["RGB", "YUV", "Y", "U", "V", "HLS", "HSV", "LUV", "LAB", "YCbCr"],
    help="color space used for accuracy calculation",
)

## Checkpointing
parser.add_argument('--resume_training', action='store_true',
                    help='resume training', default=False)
parser.add_argument('--resume_path', default='none', type=str)

## classifier
parser.add_argument('--train_classifier', default=True)
parser.add_argument('--freeze_backbone', default=True)
parser.add_argument('--num_classes', default=10)

args = parser.parse_args()


def main():
    print(args.saved_path)
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')


    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')


    main_worker(args.gpu,  args)

def main_worker(gpu,  args):

    args.gpu = gpu

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.dataset == "cifar100":
        args.num_classes = 100

    # create model
    print("=> creating cnn model '{}'".format(args.arch))
    model = set_model(args)

    torch.cuda.set_device(args.gpu)
    model = model.cuda(args.gpu)

    # Load CIFAR-10 for classifier training
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(32),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010]),
    ])

    if args.dataset == 'cifar10':
        train_dataset = torchvision.datasets.CIFAR10(root='data', train=True, download=True, transform=train_transform)
    else:
        train_dataset = torchvision.datasets.CIFAR100(root='data', train=True, download=True, transform=train_transform)
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=16)


    checkpoint_path = args.saved_path + f'last.pth.tar'

    # Load the full checkpoint
    checkpoint = torch.load(checkpoint_path)
    # Extract the backbone part of the state dictionary
    backbone_state_dict = {k.replace("backbone.", ""): v for k, v in checkpoint['state_dict'].items() if k.startswith("backbone.")}
    # Load the backbone state dict into the model's backbone
    model.backbone.load_state_dict(backbone_state_dict)

    # Define optimizer, loss, and other settings
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Training loop
    for epoch in range(args.epochs):  # Train for 10 epochs
        model.train()
        running_loss = 0.0

        correct = 0
        total = 0

        for inputs, targets in train_loader:
            inputs, targets = inputs.to(f"cuda:{args.gpu}"), targets.to(f"cuda:{args.gpu}")

            # Forward pass
            outputs = model(classifier_input=inputs)
            loss = criterion(outputs, targets)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

                        # Predictions
            _, predicted = outputs.max(1)
            correct += predicted.eq(targets).sum().item()
            total += targets.size(0)

        running_loss = running_loss / len(train_loader)
        train_acc = 100. * correct / total
        # print(f"Epoch [{epoch+1}/10], Loss: {running_loss/len(train_loader)}")
        print(f'Epoch [{epoch+1}/{args.epochs}], Loss: {running_loss:.4f}, Accuracy: {train_acc:.2f}%')

    torch.save({
    'model': model.state_dict(),
    'optimizer': optimizer.state_dict(),
    'epoch': epoch,
    'acc':train_acc,
    }, os.path.join(args.saved_path, 'classifier.pt'))
    
if __name__ == '__main__':
    main()