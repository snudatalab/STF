
import numpy as np
import pandas as pd
import torch.nn.functional as F

from train import *
from dotmap import DotMap


def online_attn_reg(factors, sfactor, rows, opts):
    ''' Implement Attention-based temporal regularization for online update '''

    mask = opts.mask
    window = opts.window
    penalty = opts.penalty
    tfactor = sfactor[window:-window, ]
    
    s_idx = make_index(window, rows)
    blocks = sfactor[s_idx, :]

    row, col = tfactor.shape

    norm = torch.norm(blocks, dim=[1,2]).view(row, 1, 1)
    blocks = blocks/norm
    cos = torch.bmm(blocks, tfactor.view(row, col, 1))
    cos = F.softmax(cos, dim=1)

    #
    target = cos[:, window]
    neighbor = cos[:, np.invert(mask)]
    blocks = blocks[:, np.invert(mask)] * neighbor
    d_it = blocks.sum(dim=1) * (1-target) * penalty
    #
    lambda_n = ((1 - target)**2).reshape(-1, 1, 1)
    
    return d_it, lambda_n

def online_time_re(factors, x_old, opts):
    '''Re-update old temporal factors'''

    mode = 2
    data = x_old.train
    rank = opts.rank
    window = opts.window
    penalty = opts.penalty
    
    tlen = data.shape[-1]
    x_re = data[:, :, -window:]
    rows = window * 2 + 1  
    
    sfactor = factors[2][tlen-window*2:tlen+window]
    
    nnz_kr, mat_b, vec_c = compute_bc(factors, x_re, mode, rank)
    
    d_it, lambda_n = online_attn_reg(factors, sfactor, window, opts)
    reg = torch.stack([torch.eye(rank)] * window) * penalty
    
    mat_b = mat_b + reg * lambda_n 
    
    vec_c = vec_c.view(-1, rank, 1) + d_it.view(-1, rank, 1)
    
    factor = torch.bmm(torch.inverse(mat_b), vec_c.view(-1, rank, 1)).sum(dim=2)
    factor = torch.where(torch.abs(factor) < 0.000001, torch.zeros_like(factor), factor)
        
    factors[-1][tlen-window:tlen] = factor


def online_time_ls(factors, x_new, opts, initialize=False):
    '''Online row-wise least square for a temporal factor'''
    
    data = x_new.train
    
    mode = 2
    size = x_new.train.shape[-1]
    rank = opts.rank
    window = opts.window
    penalty = opts.penalty
    
    if initialize:
        new_factor = gen_random((size, rank))
        sfactor = torch.cat([factors[-1][-window:, ].clone(), new_factor])
        sfactor = F.pad(sfactor.t(), (0, window)).t()
    else:
        new_factor = factors[-1][-size:, :].clone()
        sfactor = factors[-1][-(size+window):, :].clone()
        sfactor = F.pad(sfactor.t(), (0, window)).t()
    
    tmp_factors = [factors[0], factors[1], new_factor]
    nnz_kr, mat_b, vec_c = compute_bc(tmp_factors, data, mode, rank)

    blocks, s_reg = online_attn_reg(factors, sfactor, size, opts)
    reg = torch.stack([torch.eye(rank)] * size) * penalty
    
    mat_b = mat_b + reg * s_reg
    
    vec_c = vec_c.view(-1, rank, 1) + blocks.view(-1, rank, 1)

    factor = torch.bmm(torch.inverse(mat_b), vec_c.view(-1, rank, 1)).sum(dim=2)
    new_factor = torch.where(torch.abs(factor) < 0.000001, torch.zeros_like(factor), factor)
        
    add_factor(factors, new_factor)

    
def initialize_helpers(factors, data, opts):
    ''' Initialize the auxiliary variables for online update '''
    
    rank = opts.rank
    nmode = opts.nmode
    window = opts.window
    
    old_data = data[:, :, :-window]
    
    helpers = DotMap()

    helpers.B = [None] * opts.nmode
    helpers.c = [None] * opts.nmode
    
    for mode in range(2):
        _, mat_b_0, vec_c_0 = compute_bc(factors, old_data, mode, rank)
        helpers.B[mode] = mat_b_0
        helpers.c[mode] = vec_c_0
        
    return helpers


def add_factor(factors, new_factor):
    ''' Add a new temporal factor to old estimates'''
    tfactors = torch.cat([factors[-1], new_factor])
    factors[-1] = tfactors
    return factors


def online_ls(factors, x_old, x_new, opts):
    ''' Online row-wise update least squares'''
    
    rank = opts.rank
    window = opts.window
    penalty = opts.penalty
    
    
    x_re = x_old.train[:, :, -window:]
    x_skip = x_new.train[:, :, -window:]
    x_new = x_new.train[:, :, :-window]
    

    for mode in range(2):
        rows = opts.ndim[mode]
        
        mat_b_0 = opts.helpers.B[mode]
        vec_c_0 = opts.helpers.c[mode]


        _, mat_b_1, vec_c_1 = compute_bc(factors, x_re, mode, rank)
        _, mat_b_2, vec_c_2 = compute_bc(factors, x_new, mode, rank)
        _, mat_b_3, vec_c_3 = compute_bc(factors, x_skip, mode, rank)
        reg = torch.stack([torch.eye(rank)] * rows) * penalty
        
        save_mat_b = mat_b_0 + mat_b_1 + mat_b_2 
        save_vec_c = vec_c_0 + vec_c_1 + vec_c_2 
            
        mat_b = save_mat_b + mat_b_3 + reg
        vec_c = save_vec_c + vec_c_3

        factor = torch.bmm(torch.inverse(mat_b), vec_c.view(-1, rank, 1)).sum(dim=2)
        factor = torch.where(torch.abs(factor) < 0.000001, torch.zeros_like(factor), factor)

        factors[mode].data = factor        
        opts.helpers.B[mode] = save_mat_b
        opts.helpers.c[mode] = save_vec_c


