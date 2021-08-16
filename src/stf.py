
import time
from read import *
from train import *
from online_train import *

            
            
def online_update(stream_num, factors, X_old, X_new, opts):
    ''' One iteration for an online update'''
    
    start = time.time()
    inter_stream = opts.start + opts.size * (stream_num - 1)
    
    online_time_ls(factors, X_new, opts)

    online_time_re(factors, X_old, opts)

    online_ls(factors, X_old, X_new, opts)

    elapsed_time = time.time() - start
    X_old = concat(X_old, X_new)

    g_nre, g_val_nre, g_test_nre =  eval_(factors, X_old)
    l_nre, l_val_nre, l_test_nre = eval_(factors, X_new, inter_stream=inter_stream)

    return [g_val_nre, g_test_nre, elapsed_time]


def eval_(factors, dataset, inter_stream=None):
    ''' Evaluate the online model'''
    
    if not inter_stream:
        tfactor = factors[-1][inter_stream:, ]
        factors = [f.data.clone() for f in factors]
        factors[-1] = tfactor
    
    lst = []
    for dtype in ['train', 'valid', 'test']:
        data = dataset[dtype]
        rec = krprod(factors, get_indices(data)).type(torch.float64)
        vals = get_nonzeros(data)
        lst.append(fitt(vals, rec).item())
            
    return lst[0], lst[1], lst[2]



def start_stf(name, verbose=False):
    ''' Start STF on temporal stream tensors
    '''
    

    if name == 'sensor':
        #init_size = 432
        init_size = 864
    else:
        #init_size = 1008
        init_size = 504
    # init_size = 864
         

    stream_size = 1
    end = -1
    
    # Read dataset
    datasets = read_data(name)
    streams, stream_num = create_streams(datasets, init_size, end, stream_size)
    
    # setting
    if name == 'beijing':
        p = 0.01
    elif name == 'sensor':
        p = 10
    elif name == 'condition':
        p = 100
    elif name == 'radar':
        p = 10
        stream_size = 5
    else:
        p = 10
        
    stream_size = 1

    X_init = streams[0]
    opts = DotMap()
    opts.rank = 5
    opts.ndim = X_init.train.shape
    opts.nmode = len(opts.ndim)
    opts.tmode = 2
    opts.ttype = 'one'
    opts.window = 1 
    opts.penalty = p
    opts.weightf = 'attn'

    opts.start = init_size
    opts.end = end
    opts.size = stream_size
    opts.stream_num = stream_num
    
    # Step1 initialize the model
    opts.mask = masked(opts.window)

    factors = [gen_random((opts.ndim[mode], opts.rank))
                   for mode in range(opts.nmode)]

    train, valid, test = X_init.train, X_init.valid, X_init.test
    nmode, tmode = X_init.nmode, X_init.tmode

    n_iters = 300
    stop = 0
    old_nre = 1e+5

    for n_iter in range(1, n_iters+1):
        time_least_square(factors, train, 2, opts)
        for mode in range(nmode-1):
            least_square(factors, train, mode, opts)
        nre = evaluate(factors, train)
        val_nre = evaluate(factors, valid)

        if n_iter % 1 == 0 and verbose:
            print(f"Iters {n_iter} NRE | train: {nre:.3f} valid: {val_nre:.3f}")
        if old_nre <= val_nre:
            if stop >=3:
                break
        old_nre = val_nre
        
    test_nre = evaluate(factors, test)
    # To check out the initialization of model
    #with open('./out/init.txt', 'a') as f:
    #    f.write(f'{name}\t{test_nre}\n')
        
    print(f"Init NRE | train: {nre:.3f} valid: {val_nre:.3f} test: {test_nre:.3f}")

    
    # Save helper matrices for B and c
    opts.helpers = initialize_helpers(factors, train, opts)
    
    # Step3. Online update
    error_list = []
    X_old = streams[0]
    for stream_num, stream in streams.items():
        if stream_num != 0:
            X_new = stream
            error = online_update(stream_num, factors, X_old, X_new, opts)
            X_old = concat(X_old, X_new)

            error_list.append(error)
    
    val_nre = float(error_list[-1][0])
    test_nre = float(error_list[-1][1])
    time = float(error_list[-1][-1])
    
    print(f"Last NRE with {time:.3f}| train: {val_nre:.3f} valid: {test_nre:.3f}")
    df = pd.DataFrame(error_list, columns=['val_nre', 'test_nre', 'time'])
    df.to_csv(f'./out/{name}_result.txt', sep='\t')
