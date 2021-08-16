"""
Accurate Online Tensor Factorization for Temporal Tensor Streams with Missing Values (CIKM 2021)
Authors:
- Dawon Ahn (dawon@snu.ac.kr), Seoul National University
- Seyun Kim (kim79@cooper.edu), Seoul National University
- U Kang (ukang@snu.ac.kr), Seoul National University

This software may be used only for research evaluation purposes.
For other purposes (e.g., commercial), please contact the authors.
"""


import torch
import numpy as np
import tensorly as tl
import torch.nn.functional as F

RANDOM = 1


def gen_random(size):
    '''
    Generate a factor with random values
    '''
    # if len(size) >1, size must be tuple
    np.random.seed(RANDOM)
    return torch.FloatTensor(np.random.randn(size[0], size[1]))


def evaluate(factors, data, metric='nre'):
    '''Evaluate the model'''
    indices_list = get_indices(data)
    vals = get_nonzeros(data)
    
    rec = krprod(factors, indices_list).type(torch.float64)
    if metric == 'nre':
        return fitt(vals, rec).item()
    else:
        return rmse(vals, rec).item()
    
def fitt(val, rec):
    '''Calculate Normalized Reconstruction Error (NRE)'''
    return torch.norm(val-rec)/torch.norm(val)

def get_nonzeros(data):
    '''Return nonzeros' value'''
    idxs = torch.nonzero(data, as_tuple=True)
    nonzeros = data[idxs]
    return nonzeros

def get_indices(data):
    '''Return nonzeros' indices'''
    idxs = torch.nonzero(data).T
    return idxs

def masked(window):
    center = window
    array = [False] * (window * 2 + 1)
    array[center] = True
    return array

def make_index(window, length):
    '''
    Make a list having neighbor indices
    '''
    
    window = window * 2 + 1
    return np.arange(0, window)[None, :] + np.arange(length)[:, None]

def krprod(factors, indices_list, skip=None):
    ''' Khatri-rao product for given nonzeros' indices 
         indices_list: nonzeros indices list having len(mode)
         factors: factor matrcies having len(mode)
         data: dense tensor
         skip: mode(s) to skip '''

    if skip is not None:
        if type(skip) != list:
            skip = [skip]
        nmode = len(factors)
        factors = [factors[mode] for mode in range(nmode) if mode not in skip]
        indices_list = [indices_list[mode] for mode in range(nmode) if mode not in skip]

    rank = tl.shape(factors[0])[1]
    nnz = len(indices_list[0])
    nnz_krprod = torch.ones((nnz, rank))

    # Compute the Khatri-Rao product for nonzeros' indices
    for indices, factor in zip(indices_list, factors):
        nnz_krprod = nnz_krprod * factor[indices, :]

    if skip is not None:
        return nnz_krprod
    else:
        return torch.sum(nnz_krprod, dim=1)

    
def compute_bc(factors, data, mode, rank):
    '''Compute intermediate data mat_b & vec_c with nnz_kr '''

    indices_list = get_indices(data)
    tmode_idx = indices_list[mode]
    vals = get_nonzeros(data)
    
    rows = data.shape[mode]

    # Compute a delta
    nnz_kr = krprod(factors, indices_list, skip=mode)

    # Compute intermediate data B and c
    mat = torch.bmm(nnz_kr.view(-1, rank, 1), nnz_kr.view(-1, 1, rank))
    mat_b = torch.zeros((rows, rank, rank), dtype=torch.float)
    mat_b = mat_b.scatter_add(0, tmode_idx.view(-1, 1, 1).expand(-1, rank, rank), mat.float())

    vec = nnz_kr * vals.view(-1, 1)
    vec_c = torch.zeros((rows, rank), dtype=torch.float)
    vec_c = vec_c.scatter_add(0, tmode_idx.view(-1, 1).expand(-1, rank), vec.float())

    return nnz_kr, mat_b, vec_c


def least_square(factors, data, mode, opts):
    '''
    Calculate least square equation for a non-temporal factor
    '''
    
    rank = opts.rank
    penalty = opts.penalty
    rows = factors[mode].shape[0]
    
    nnz_kr, mat_b, vec_c = compute_bc(factors, data, mode, rank)
    
    # Regularization
    reg = torch.stack([torch.eye(rank)] * rows) * penalty
    mat_b = mat_b + reg 
        
    # Updating factors
    factor = torch.bmm(torch.inverse(mat_b), vec_c.view(-1, rank, 1)).sum(dim=2)
    factor = torch.where(torch.abs(factor) < 0.000001, torch.zeros_like(factor), factor)
    factors[mode].data = factor



def time_least_square(factors, data, mode, opts):
    '''
    Calculate least square equation for a temporal factor
    '''
    
    rank = opts.rank
    penalty = opts.penalty
    rows = factors[mode].shape[0]
    
    nnz_kr, mat_b, vec_c = compute_bc(factors, data, mode, rank)
    
    # Select smoothing function
    blocks, s_reg = attn_reg(factors, rows, opts)
 
    # Regularization
    reg = torch.stack([torch.eye(rank)] * rows) * penalty

    mat_b = mat_b + reg * s_reg 

    vec_c = vec_c.view(-1, rank, 1) + blocks.view(-1, rank, 1)
        
    factor = torch.bmm(torch.inverse(mat_b), vec_c.view(-1, rank, 1)).sum(dim=2)
    factor = torch.where(torch.abs(factor) < 0.000001, torch.zeros_like(factor), factor)
    
    factors[mode].data = factor



def attn_reg(factors, rows, opts):
    ''' Attention-based temporal regularization for a static setting
    '''
    
    mask = opts.mask
    window = opts.window
    penalty = opts.penalty

    tmode = opts.tmode
    tfactor = factors[tmode]
    
    s_idx = make_index(window, rows)
    
    pad_data = F.pad(tfactor.t(), (window, window))
    pad_data = pad_data.t()
    blocks = pad_data[s_idx, :]

    row, col = tfactor.shape
    norm = torch.norm(blocks, dim=[1,2]).view(row, 1, 1)
    blocks = blocks/norm
    cos = torch.bmm(blocks, tfactor.view(row, col, 1))
    cos = F.softmax(cos, dim=1)

    #
    target = cos[:, window]
    neighbor = cos[:, np.invert(mask)]
    blocks = blocks[:, np.invert(mask)] * neighbor
    blocks = blocks.sum(dim=1) * (1-target) * penalty

    reg = ((1 - target)**2).reshape(-1, 1, 1)

    return blocks, reg


