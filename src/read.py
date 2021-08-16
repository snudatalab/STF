
import os
import torch
import numpy as np
import pandas as pd
import tensorly as tl

from dotmap import DotMap
from tensorly import cp_to_tensor
from sklearn.model_selection import train_test_split


def size(y):
    return y.shape

def read_data(name='chicago', path='data'):
    '''
    Read dataset and split it into train, valid, test dataset
    '''
    
    data = f'{path}/{name}.tensor'
    df = pd.read_csv(data, sep = '\t', header=None)
    df = df[[1, 2, 0, 3]]

    
    if name == 'chicago':
        df = df[df[0] != 2904]
        df[3] = np.log2(df[3] + 1)
    if name == 'condition':
        df = df[df[0] <= 2622]
    if name == 'beijing':
        df[3] = np.log2(df[3] + 1)
        df = df[df[0] <= 5994]
    if name == 'madrid':
        df[3] = np.log2(df[3] + 1)
        
    X_train, tmp = train_test_split(df, test_size=0.2, random_state=1, shuffle=True)
    X_valid, X_test = train_test_split(tmp, test_size=0.5, random_state=1, shuffle=True)
    
    X_train = make_tensor(X_train)
    X_valid = make_tensor(X_valid)
    X_test = make_tensor(X_test)
    
    if (size(X_train) != size(X_valid)) and (size(X_valid) != size(X_test)):
        print("Size shape error", size(X_train) , size(X_valid), size(X_test))
        exit()
    
    datasets = DotMap()
    
    datasets.train = X_train
    datasets.valid = X_valid
    datasets.test = X_test
    
    datasets.tmode = 2
    datasets.ndim = X_train.shape
    datasets.nmode = len(datasets.ndim)
    
    return datasets


def make_tensor(df):
    '''
    Make COO format file into a tensor
    '''
    
    indices = df.iloc[:, :-1].values
    values = df.iloc[:, -1].values

    stensor = torch.sparse_coo_tensor(indices.T, values)
    dtensor = stensor.to_dense()
    
    return dtensor


def create_streams(X, start, end, size):
    '''Create tensor streams with init size, entire size, stream size'''
    
    if end == -1:
        total_size = X.train.shape[-1]
    else:
        total_size = end
    stream_num = np.ceil((total_size - start) / size).astype(int)
    
    start_ = 0
    end_ = start
    streams = DotMap()
    for i in range(stream_num+1):
        streams[i] = DotMap()
        streams[i].train = X.train[:, :, start_:end_]
        streams[i].valid = X.valid[:, :, start_:end_]
        streams[i].test = X.test[:, :, start_:end_]
        streams[i].tmode = 2
        streams[i].ndim = streams[i].train.shape
        streams[i].nmode = len(streams[i].ndim)
        start_ = end_
        end_ += size
    
    return streams, stream_num


def concat(a, b):
    '''Concatenate old streams wit a new stream'''
    
    new = DotMap()
    
    new.train = torch.cat((a.train, b.train), 2)
    new.valid = torch.cat((a.valid, b.valid), 2)
    new.test = torch.cat((a.test, b.test), 2)
    
    new.tmode = 2
    new.ndim = new.train.shape
    new.nmode = len(new.ndim)
    
    return new
    



