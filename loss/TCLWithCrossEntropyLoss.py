""" Triplet-Centor Loss + CrossEntropy Loss
"""

import torch.nn as nn

class TCLWithCrossEntropyLoss(nn.Module):
    def __init__(self):
        super(TCLWithCrossEntropyLoss, self).__init__()
