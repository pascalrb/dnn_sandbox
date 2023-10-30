import numpy as np
from sklearn.cluster import KMeans
from pruned_layers import *
import torch.nn as nn

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def quantize_whole_model(net, bits=8):
    """
    Quantize the whole model.
    :param net: (object) network model.
    :return: centroids of each weight layer, used in the quantization codebook.
    """
    cluster_centers = []  
        #[[centroids of PrunedLay1], [centroids of PrunedLay2], ...]

    assert isinstance(net, nn.Module)
    layer_ind = 0
    num_of_clusters = 2 ** bits
    for n, m in net.named_modules():
        if isinstance(m, PrunedConv):
            """
            Applying quantization for the PrunedConv layer.
            """
            weight = m.conv.weight.flatten().reshape(-1,1)
            kmeans = KMeans(n_clusters=num_of_clusters).fit(weight.detach().cpu().numpy())
            centroids = kmeans.cluster_centers_.flatten()
            cluster_centers.append(centroids)

            # Resave weights as cluster indices
            m.conv.weight.data = torch.tensor(kmeans.labels_, dtype=m.conv.weight.dtype).reshape_as(m.conv.weight)
            m.is_quantized = True

            layer_ind += 1
            print("Complete %d layers quantization (Conv)..." %layer_ind)

        elif isinstance(m, PruneLinear):
            """
            Applying quantization for the PrunedLinear layer.
            """
            weight = m.linear.weight.flatten().reshape(-1,1)
            kmeans = KMeans(n_clusters=num_of_clusters).fit(weight.detach().cpu().numpy())
            centroids = kmeans.cluster_centers_.flatten()
            cluster_centers.append(centroids)

            # Resave weights as cluster indices
            m.linear.weight.data = torch.tensor(kmeans.labels_, dtype=m.linear.weight.dtype).reshape_as(m.linear.weight)
            m.is_quantized = True

            layer_ind += 1
            print("Complete %d layers quantization (FC)..." %layer_ind)

    return np.array(cluster_centers)
