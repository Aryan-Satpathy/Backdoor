import torch
from torch import nn
import torch.nn.functional as F 
import torchvision.transforms as T
import copy
from methods.base import CLModel
from networks.resnet_org import model_dict
from networks.resnet_cifar import model_dict as model_dict_cifar
from .losses import infonce_loss


class MoCoV3(CLModel):
    """ 
    MoCo v3: Momentum Contrast v3
    Link: https://arxiv.org/abs/2104.02057
    Implementation: https://github.com/facebookresearch/moco-v3
    """
    # def __init__(self, backbone, feature_size, projection_dim=256, hidden_dim=2048, temperature=0.5, m=0.999,
    #              image_size=32):
    #     super().__init__()

    def __init__(self, args):
      
        super(MoCoV3, self).__init__(args)

        self.temperature = args.temp
        self.criterion = infonce_loss
        self.m = args.moco_m
        self.backbone =  self.model_generator()
        self.projector = Projector(self.feat_dim, hidden_dim=2048, out_dim=256)
        #self.image_size = image_size
        self.encoder_q = self.encoder = nn.Sequential(self.backbone, self.projector)
        self.predictor = Predictor(in_dim=256, hidden_dim=2048, out_dim=256)
        self.encoder_k = copy.deepcopy(self.encoder_q)
        self._init_encoder_k()

        
    def forward(self, x1, x2):
        """
        Input:
            x1: first views of images
            x2: second views of images
        """


        q1 = self.predictor(self.encoder_q(x1))
        q2 = self.predictor(self.encoder_q(x2))

        with torch.no_grad():
            
            self._update_momentum_encoder()
            k1 = self.encoder_k(x1)
            k2 = self.encoder_k(x2)
        
        return q1,q2,k1,k2
        # loss = infonce_loss(q1, k2, self.temperature) + infonce_loss(q2, k1, self.temperature)
        # return loss
        
    @torch.no_grad()
    def _update_momentum_encoder(self):
        for param_b, param_m in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_m.data = param_m.data * self.m + param_b.data * (1. - self.m)
            
    @torch.no_grad()
    def _init_encoder_k(self):
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data) 
            param_k.requires_grad = False 
         




class Projector(nn.Module):
    """ Projector for SimCLR v2, used in MoCo v3 too """
    def __init__(self, in_dim, hidden_dim=2048, out_dim=256):
        super().__init__()
        
        self.layer1 = nn.Sequential(
                    nn.Linear(in_dim, hidden_dim),
                    nn.BatchNorm1d(hidden_dim, eps=1e-5, affine=True),
                    nn.ReLU(inplace=True),
                    )
        self.layer2 = nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.BatchNorm1d(hidden_dim, eps=1e-5, affine=True),
                    nn.ReLU(inplace=True),
                    )
        self.layer3 = nn.Sequential(
                    nn.Linear(hidden_dim, out_dim),
                    nn.BatchNorm1d(out_dim, eps=1e-5, affine=True),
                    )
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x 
    
    
class Predictor(nn.Module):
    """ Projection Head and Prediction Head for BYOL, used in MoCo v3 too """
    def __init__(self, in_dim, hidden_dim=4096, out_dim=256):
        super().__init__()

        self.layer1 = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True)
        )
        self.layer2 = nn.Sequential(
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        return x       
    
    
