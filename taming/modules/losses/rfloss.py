import torch
import torch.nn as nn
import torch.nn.functional as F


class RFDATATLoss(nn.Module):
    def __init__(self):
        super().__init__()

    
    def forward(self, target, logits):
        codebook_loss = torch.mean((target.detach()-logits)**2)

        return codebook_loss.mean()