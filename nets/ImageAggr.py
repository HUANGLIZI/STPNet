import os
import torch
import torch.nn as nn
import numpy as np
from transformers import BertModel
import logging
from transformers import AutoModel, AutoTokenizer
from .mlp import FC_MLP

logger = logging.getLogger(__name__)


# 'True' represents to be masked （Do not participate in the calculation of attention）
# 'False' represents not to be masked
def padding_mask(embs, lengths):

    mask = torch.ones(len(lengths), embs.shape[1], device=lengths.device)
    for i in range(mask.shape[0]):
        end = int(lengths[i])
        mask[i, :end] = 0.

    return mask.bool()


def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.)


def l2norm(X, dim, eps=1e-8):
    norm = torch.pow(X, 2).sum(dim=dim, keepdim=True).sqrt() + eps
    X = torch.div(X, norm)
    return X


def maxk_pool1d(x, dim, k):
    max_k = maxk(x, dim, k)
    return max_k.mean(dim)


def maxk(x, dim, k):
    _x, index = x.topk(k, dim=dim)
    return _x

    
# uncertain length
def maxk_pool1d_var(x, dim, k, lengths):
    # k >= 1
    results = []
    # assert len(lengths) == x.size(0)

    for idx in range(x.size(0)):
        # keep use all number of features
        k = min(k, int(lengths[idx].item()))

        tmp = torch.split(x[idx], split_size_or_sections=lengths[idx], dim=dim-1)[0]

        max_k_i = maxk_pool1d(tmp, dim-1, k)
        results.append(max_k_i)

    # construct with the batch
    results = torch.stack(results, dim=0)

    return results


def avg_pool1d_var(x, dim, lengths):

    results = []
    # assert len(lengths) == x.size(0)

    for idx in range(x.size(0)):

        # keep use all number of features
        tmp = torch.split(x[idx], split_size_or_sections=lengths[idx], dim=dim-1)[0]
        avg_i = tmp.mean(dim-1)

        results.append(avg_i)

    # construct with the batch
    results = torch.stack(results, dim=0)

    return results


class Maxk_Pooling(nn.Module):
    def __init__(self, dim=1, k=2):
        super(Maxk_Pooling, self).__init__()

        self.dim = dim
        self.k = k

    def forward(self, features, lengths):

        pool_weights = None
        pooled_features = maxk_pool1d_var(features, dim=self.dim, k=self.k, lengths=lengths)
        
        return pooled_features, pool_weights


class EncoderImageAggr(nn.Module):
    def __init__(self, img_dim=2048, embed_size=1024):
        super(EncoderImageAggr, self).__init__()

        self.embed_size = embed_size
        
        self.fc = FC_MLP(img_dim, embed_size // 2, embed_size, 2, bn=True)           
        self.fc.apply(init_weights)

        self.maxpool = Maxk_Pooling()

    def forward(self, images, image_lengths):

        img_emb = self.fc(images)
        img_emb_res, _ = self.maxpool(img_emb, image_lengths)

        return img_emb_res




if __name__ == '__main__':

    pass