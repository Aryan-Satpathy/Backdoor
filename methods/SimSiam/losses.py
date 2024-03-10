import torch.nn.functional as F

def negative_cosine_similarity(p1,p2, z1,z2):
    """ Negative Cosine Similarity """
    z1 = z1.detach()
    p1 = F.normalize(p1, dim=1)
    z1 = F.normalize(z1, dim=1)
    z2 = z2.detach()
    p2 = F.normalize(p2, dim=1)
    z2 = F.normalize(z2, dim=1)
    return -(p1 * z2).sum(dim=1).mean() / 2 -(p2 * z1).sum(dim=1).mean() / 2



