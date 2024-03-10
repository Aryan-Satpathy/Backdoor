import torch
import torch.nn as nn
import torch.nn.functional as F

# def infonce_loss(q, k, temperature=0.07):
#     """ InfoNCE loss """
#     q = nn.functional.normalize(q, dim=1)
#     k = nn.functional.normalize(k, dim=1)
#     logits = torch.einsum('nc,mc->nm', [q, k])
#     logits /= temperature
#     labels = (torch.arange(logits.shape[0], dtype=torch.long)).to(q.device)
#     return F.cross_entropy(logits, labels)

def infonce_loss(q1, q2, k1, k2, temperature=0.07):
    """ InfoNCE loss """
    q1 = nn.functional.normalize(q1, dim=1)
    k2 = nn.functional.normalize(k2, dim=1)
    logits1 = torch.einsum('nc,mc->nm', [q1, k2])
    logits1 /= temperature
    labels1 = (torch.arange(logits1.shape[0], dtype=torch.long)).to(q1.device)
    
    q2 = nn.functional.normalize(q2, dim=1)
    k1 = nn.functional.normalize(k1, dim=1)
    logits2 = torch.einsum('nc,mc->nm', [q2, k1])
    logits2 /= temperature
    labels2 = (torch.arange(logits2.shape[0], dtype=torch.long)).to(q2.device)
    
    sum_loss = F.cross_entropy(logits1, labels1) + F.cross_entropy(logits2, labels2)
    #print(sum_loss)
    return sum_loss

 

