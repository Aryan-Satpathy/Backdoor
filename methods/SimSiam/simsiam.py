
import torch
import torch.nn as nn
import torch.nn.functional as F
from methods.base import CLModel
from .losses import negative_cosine_similarity


class SimSiam(CLModel):
    """ 
    SimSiam: Exploring Simple Siamese Representation Learning
    Link: https://arxiv.org/abs/2011.10566
    Implementation: https://github.com/facebookresearch/simsiam
    """
    # def __init__(self,  projection_dim=2048, hidden_dim_proj=2048, hidden_dim_pred=512,
    #              ):
    #     super().__init__()

    def __init__(self, args):
         
        super(SimSiam, self).__init__(args)

        
        self.criterion = negative_cosine_similarity
        self.backbone = self.model_generator()
        self.projector = Projector(self.feat_dim, hidden_dim=2048, out_dim=256)
        self.predictor = Predictor(in_dim=256, hidden_dim=512, out_dim=256)
        self.encoder = nn.Sequential(self.backbone, self.projector)

        # Classifier head for training after loading backbone
        if args.train_classifier:
            # Create a classifier head on top of the backbone features
            self.classifier_head = nn.Linear(self.feat_dim, args.num_classes)

        # Option to freeze or unfreeze backbone during classifier training
        self.freeze_backbone = args.freeze_backbone
        if args.freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
       
        
    def forward(self, x1=None, x2=None, classifier_input=None):
        if classifier_input is not None:
            # Forward pass through the backbone and classifier head
            x = self.backbone(classifier_input)
            logits = self.classifier_head(x)
            return logits
        else:
            z1, z2 = self.encoder(x1), self.encoder(x2) 
            p1, p2 = self.predictor(z1), self.predictor(z2)
            return p1, p2, z1, z2



class Projector(nn.Module):
    """ Projection Head for SimSiam """
    def __init__(self, in_dim, hidden_dim=2048, out_dim=2048):
        super().__init__()

        self.layer1 = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True)
        )
        self.layer2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True)
        )
        self.layer3 = nn.Sequential(
            nn.Linear(hidden_dim, out_dim),
            nn.BatchNorm1d(out_dim, affine=False)
        )

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x 
    
    
    
class Predictor(nn.Module):
    """ Predictor for SimSiam """
    def __init__(self, in_dim=2048, hidden_dim=512, out_dim=2048):
        super().__init__()
        
        self.layer1 = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True)
        )
        self.layer2 = nn.Linear(hidden_dim, out_dim)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        return x 
