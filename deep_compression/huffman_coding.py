import numpy as np
from sklearn.cluster import KMeans
from pruned_layers import *
import torch.nn as nn

from collections import Counter
import heapq

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Huffman coding algo inspired by GeeksForGeeks' implementation
class Node: 
   def __init__(self, w_quant, freq, left=None, right=None): 
       # frequency of w_quant 
       self.freq = freq 
 
       # w_quant name (quantized weight (a centroid from list of centroids)) 
       self.w_quant = w_quant 
 
       # node left of current node 
       self.left = left 
 
       # node right of current node 
       self.right = right 
 
       # tree direction (0/1) 
       self.huff = '' 
 
   def __lt__(self, nxt): 
       return self.freq < nxt.freq 

def _huffman_coding_algo(frequency):

    ENCODINGS = {}

    def assign_encoding_dict(node, val=''):
        new_val = val + node.huff

        if(node.left):
            assign_encoding_dict(node.left, new_val)
        if(node.right):
            assign_encoding_dict(node.right, new_val)

        if(not node.left and not node.right):
            ENCODINGS[node.w_quant] = new_val 
        

    nodes = []
    
    for key, val in frequency.items():
        heapq.heappush(nodes, Node(key, val))

    while len(nodes) > 1:
        left = heapq.heappop(nodes)
        right = heapq.heappop(nodes)

        left.huff = '0'
        right.huff = '1'

        new_node = Node(left.w_quant+right.w_quant, left.freq+right.freq, left, right)

        heapq.heappush(nodes, new_node)

    assign_encoding_dict(nodes[0])

    return ENCODINGS


def _huffman_coding_per_layer(weight, centers):
    """
    Huffman coding at each layer
    :param weight: weight parameter of the current layer.
    :param centers: KMeans centroids in the quantization codebook of the current weight layer.
    :return: 
            'encodings': Encoding map mapping each weight parameter to its Huffman coding.
            'frequency': Frequency map mapping each weight parameter to the total number of its appearance.
            'encodings' should be in this format:
            {"0.24315": '0', "-0.2145": "100", "1.1234e-5": "101", ...
            }
            'frequency' should be in this format:
            {"0.25235": 100, "-0.2145": 42, "1.1234e-5": 36, ...
            }
            'encodings' and 'frequency' does not need to be ordered in any way.
    """
    """
    Generate Huffman Coding and Frequency Map according to incoming weights and centers (KMeans centriods).
    """
    frequency = {}
    encodings = {}

    ''' 
    Note: This Huffman coding function expects the model to have been qunatized 
        thus expects qunatized weights as parameters.
    '''

    for w in weight.flatten():
        # Weight contains cluster indices of quantized weights
        w_quant = str(centers[int(w)])
        frequency[w_quant] = frequency.get(w_quant, 0) + 1

    encodings = _huffman_coding_algo(frequency)
    
    return encodings, frequency


def compute_average_bits(encodings, frequency):
    """
    Compute the average storage bits of the current layer after Huffman Coding.
    :param 'encodings': Encoding map mapping each weight parameter to its Huffman coding.
    :param 'frequency': Frequency map mapping each weight parameter to the total number of its appearance.
            'encodings' should be in this format:
            {"0.24315": '0', "-0.2145": "100", "1.1234e-5": "101", ...
            }
            'frequency' should be in this format:
            {"0.25235": 100, "-0.2145": 42, "1.1234e-5": 36, ...
            }
            'encodings' and 'frequency' does not need to be ordered in any way.
    :return (float) a floating value represents the average bits.
    """
    #total = 0
    total = 1
    total_bits = 0
    for key in frequency.keys():
        total += frequency[key]
        total_bits += frequency[key] * len(encodings[key])
    return total_bits / total

def huffman_coding(net, centers):
    """
    Apply huffman coding on a 'quantized' model to save further computation cost.
    :param net: a 'nn.Module' network object.
    :param centers: KMeans centroids in the quantization codebook for Huffman coding.
    :return: frequency map and encoding map of the whole 'net' object.
    """
    assert isinstance(net, nn.Module)
    layer_ind = 0
    freq_map = []
    encodings_map = []
    i = 0
    avg_avg = 0
    for n, m in net.named_modules():
        if isinstance(m, PrunedConv):
            i+=1
            weight = m.conv.weight.data.cpu().numpy()
            center = centers[layer_ind]
            orginal_avg_bits = round(np.log2(len(center)))
            print("\nOriginal storage for each parameter: %.4f bits" %orginal_avg_bits)
            encodings, frequency = _huffman_coding_per_layer(weight, center)
            freq_map.append(frequency)
            encodings_map.append(encodings)
            huffman_avg_bits = compute_average_bits(encodings, frequency)
            avg_avg += huffman_avg_bits
            print("Average storage for each parameter after Huffman Coding: %.4f bits" %huffman_avg_bits)
            layer_ind += 1
            print("Complete %d layers for Huffman Coding (Conv)..." %layer_ind)
        elif isinstance(m, PruneLinear):
            i+=1
            weight = m.linear.weight.data.cpu().numpy()
            center = centers[layer_ind]
            orginal_avg_bits = round(np.log2(len(center)))
            print("\nOriginal storage for each parameter: %.4f bits" %orginal_avg_bits)
            encodings, frequency = _huffman_coding_per_layer(weight, center)
            freq_map.append(frequency)
            encodings_map.append(encodings)
            huffman_avg_bits = compute_average_bits(encodings, frequency)
            avg_avg += huffman_avg_bits
            print("Average storage for each parameter after Huffman Coding: %.4f bits" %huffman_avg_bits)
            layer_ind += 1
            print("Complete %d layers for Huffman Coding (FC)..." %layer_ind)

    avg_avg = avg_avg/i
    print(f'\navg_avg: {avg_avg}')

    return freq_map, encodings_map
