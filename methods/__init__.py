from .SimCLR.simclr import SimCLRModel
from .BYOL.byol import BYOL
from .MoCo.mocov3 import MoCoV3
from .SimSiam.simsiam import SimSiam

def set_model(args):
    if args.method == 'simclr':
        return SimCLRModel(args)
    
    elif args.method == 'byol':
        return BYOL(args)
    
    elif args.method=='moco':
        return MoCoV3(args)

    elif args.method=='simsiam':
        return SimSiam(args)

    else:
        raise  NotImplementedError

