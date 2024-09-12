import os 
import sys 
import time 
import torch 
import torch.nn as nn 
import torch.nn.functional as F 
import torch.optim as optim

from warmup_scheduler import GradualWarmupScheduler

from methods.base import CLModel, CLTrainer
from .losses import SupConLoss
from utils.util import AverageMeter, save_model, load_model
from utils.knn import knn_monitor 

class SimCLRModel(CLModel):
    def __init__(self, args):
        super().__init__(args)
        # self.criterion = SupConLoss(self.args.temp).cuda(self.args.gpu)
        self.criterion = SupConLoss(args.temp).cuda()

        if self.mlp_layers == 2:
            self.proj_head = nn.Sequential(
                    nn.Linear(self.feat_dim, self.feat_dim),
                    nn.ReLU(inplace=True),
                    nn.Linear(self.feat_dim, 128)
                )
        elif self.mlp_layers == 3:
            self.proj_head = nn.Sequential(
                    nn.Linear(self.feat_dim, self.feat_dim),
                    nn.ReLU(inplace=True),
                    nn.Linear(self.feat_dim, self.feat_dim),
                    nn.ReLU(inplace=True),
                    nn.Linear(self.feat_dim, 128)
                )
        
        # Classifier head for training after loading backbone
        if args.train_classifier:
            # Create a classifier head on top of the backbone features
            self.classifier_head = nn.Linear(self.feat_dim, args.num_classes)

        # Option to freeze or unfreeze backbone during classifier training
        self.freeze_backbone = args.freeze_backbone
        if args.freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False


    @torch.no_grad()
    def moving_average(self):
        """
        Momentum update of the key encoder
        """
        m = 0.5
        for param_q, param_k in zip(self.distill_backbone.parameters(), self.backbone.parameters()):
            param_k.data = param_k.data * m + param_q.data * (1. - m)
        
    def forward(self, v1=None, v2=None, classifier_input=None):
        if classifier_input is not None:
            # Forward pass through the backbone and classifier head
            x = self.backbone(classifier_input)
            logits = self.classifier_head(x)
            return logits
        else:
            x = torch.cat([v1, v2], dim=0)
            x = self.backbone(x)
            reps = F.normalize(self.proj_head(x), dim=1)

            bsz = reps.shape[0] // 2
            f1, f2 = torch.split(reps, [bsz, bsz], dim=0)
            features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)

            return features

        
