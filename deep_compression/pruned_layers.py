import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

# built-in PyTorch pruning
import torch.nn.utils.prune as tprune

device = "cuda" if torch.cuda.is_available() else "cpu"
        
class PruneLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super(PruneLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.linear = nn.Linear(in_features, out_features)
        self.mask = torch.ones_like(self.linear.weight.data)
        m = self.in_features
        n = self.out_features
        self.is_quantized = False

        # Initailization 
        self.sparsity = 1.0
        self.linear.weight.data.normal_(0, math.sqrt(2. / (m+n)))

    def forward(self, x):
        out = self.linear(x)
        return out

    def prune_by_percentage(self, q=5.0):
        """
        Pruning the weight paramters by threshold.
        :param q: pruning percentile. 'q' percent of the least 
        significant weight parameters will be pruned.
        """
        """
        Prune the weight connections by percentage. Calculate the sparisty after 
        pruning and store it into 'self.sparsity'.
        Store the pruning pattern in 'self.mask' for further fine-tuning process 
        with pruned connections.
        """
        if(q <= 1):
            q = int(q * torch.numel(self.linear.weight))
        topk = torch.topk(torch.abs(self.linear.weight).view(-1), k=q, largest=False)
        self.mask.view(-1)[topk.indices] = 0
        self.linear.weight.data = self.linear.weight.to(device) * self.mask.to(device)

        weight = self.linear.weight.flatten()
        num_parameters = weight.shape[0]
        num_nonzero_parameters = (weight != 0).sum()
        self.sparsity = 1 - num_nonzero_parameters / num_parameters

    def prune_by_std(self, s=0.25):
        """
        Pruning by a factor of the standard deviation value.
        :param std: (scalar) factor of the standard deviation value. 
        Weight magnitude below np.std(weight)*std
        will be pruned.
        """

        """
        Prune the weight connections by standarad deviation. 
        Calculate the sparisty after pruning and store it into 'self.sparsity'.
        Store the pruning pattern in 'self.mask' for further fine-tuning process 
        with pruned connections.
        """
        threshold = torch.std(self.linear.weight) * s

        mask_l = torch.le(self.linear.weight, -threshold).long()
        mask_r = torch.ge(self.linear.weight, threshold).long()
        self.mask = torch.bitwise_or(mask_l, mask_r)
        self.linear.weight.data = self.linear.weight * self.mask

        num_parameters = torch.numel(self.linear.weight)
        num_nonzero_parameters = torch.count_nonzero(self.linear.weight)
        self.sparsity = 1 - num_nonzero_parameters / num_parameters

class PrunedConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=False):
        super(PrunedConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)
        self.is_quantized = False

        # Expand and Transpose to match the dimension
        self.mask = torch.ones_like(self.conv.weight.data)

        # Initialization
        n = self.kernel_size * self.kernel_size * self.out_channels
        m = self.kernel_size * self.kernel_size * self.in_channels
        self.conv.weight.data.normal_(0, math.sqrt(2. / (n+m) ))
        self.sparsity = 1.0

    def forward(self, x):
        out = self.conv(x)
        return out

    def prune_by_percentage(self, q=5.0):
        """
        Pruning the weight paramters by threshold.
        :param q: pruning percentile. 'q' percent of the least 
        significant weight parameters will be pruned.
        """
        """
        Prune the weight connections by percentage. Calculate the sparisty after 
        pruning and store it into 'self.sparsity'.
        Store the pruning pattern in 'self.mask' for further fine-tuning process 
        with pruned connections.
        """
        if(q <= 1):
            q = int(q * torch.numel(self.conv.weight))
        topk = torch.topk(torch.abs(self.conv.weight).view(-1), k=q, largest=False)
        self.mask.view(-1)[topk.indices] = 0
        self.conv.weight.data = self.conv.weight.to(device) * self.mask.to(device)

        weight = self.conv.weight.flatten()
        num_parameters = weight.shape[0]
        num_nonzero_parameters = (weight != 0).sum()
        self.sparsity = 1 - num_nonzero_parameters / num_parameters

    def prune_by_std(self, s=0.25):
        """
        Pruning by a factor of the standard deviation value.
        :param s: (scalar) factor of the standard deviation value. 
        Weight magnitude below np.std(weight)*std
        will be pruned.
        """
        
        """
        Prune the weight connections by standarad deviation. 
        Calculate the sparisty after pruning and store it into 'self.sparsity'.
        Store the pruning pattern in 'self.mask' for further fine-tuning process 
        with pruned connections.
        """
        threshold = torch.std(self.conv.weight) * s

        mask_l = torch.le(self.conv.weight, -threshold).long()
        mask_r = torch.ge(self.conv.weight, threshold).long()
        self.mask = torch.bitwise_or(mask_l, mask_r)
        self.conv.weight.data = self.conv.weight * self.mask

        num_parameters = torch.numel(self.conv.weight)
        num_nonzero_parameters = torch.count_nonzero(self.conv.weight)
        self.sparsity = 1 - num_nonzero_parameters / num_parameters



"""
Custom pruning class extending PyTorch's BasedPruningMethod to prune 
by standard deviation
"""
class STDUnstructured(tprune.BasePruningMethod):
    PRUNING_TYPE = "unstructured"

    def __init__(self, std_fact=0.25):
        self.std_fact = std_fact

    def compute_mask(self, t, default_mask):
        mask = default_mask.clone()

        std = torch.std(t) * self.std_fact

        mask_l = torch.le(t, -std).long()
        mask_r = torch.ge(t, std).long()
        mask = torch.bitwise_or(torch.bitwise_or(mask_l, mask_r), mask)

        return mask

    @classmethod
    def apply(cls, module, name, std_fact, importance_scores=None):
        return super().apply(module, name, std_fact=std_fact, importance_scores=importance_scores)

def std_unstructured(module, name, std_fact, importance_scores=None):
    STDUnstructured.apply(module, name, std_fact=std_fact, importance_scores=importance_scores)
    return module