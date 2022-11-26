import scipy.io as sio                     # import scipy.io for .mat file I/
import numpy as np                         # import numpy

# Implementated based on the PyTorch
import DDPG_unmodified.ddpg as ddpgy
from optimization import bisection

import time


def plot_rate(rate_his, rolling_intv=50):
    import matplotlib.pyplot as plt
    import pandas as pd
    import matplotlib as mpl

    rate_array = np.asarray(rate_his)
    df = pd.DataFrame(rate_his)


    mpl.style.use('seaborn')
    fig, ax = plt.subplots(figsize=(15, 8))
#    rolling_intv = 20

    plt.plot(np.arange(len(rate_array))+1, np.hstack(df.rolling(rolling_intv, min_periods=1).mean().values), 'b')
    plt.fill_between(np.arange(len(rate_array))+1, np.hstack(df.rolling(rolling_intv, min_periods=1).min()[0].values), np.hstack(df.rolling(rolling_intv, min_periods=1).max()[0].values), color = 'b', alpha = 0.2)
    plt.ylabel('Normalized Computation Rate')
    plt.xlabel('Time Frames')
    plt.show()

def save_to_txt(rate_his, file_path):
    with open(file_path, 'w') as f:
        for rate in rate_his:
            f.write("%s \n" % rate)

if __name__ == "__main__":
    '''
        This algorithm generates K modes from DNN, and chooses with largest
        reward. The mode with largest reward is stored in the memory, which is
        further used to train the DNN.
        Adaptive K is implemented. K = max(K, K_his[-memory_size])
    '''

    REPLACEMENT = [
        dict(name='soft', tau=0.005),
        dict(name='hard', rep_iter=600)
    ][0]  # you can try different target replacement strategies



    N = 10                       # number of users
    n = 30000                    # number of time frames
    decoder_mode = 'OP'          # the quantization mode could be 'OP' (Order-preserving) or 'KNN'
    Memory = 1024                # capacity of memory structure

    print('#user = %d, #channel=%d, decoder = %s, Memory = %d'%(N,n,decoder_mode, Memory))
    # Load data
    channel = sio.loadmat('/home/yuhang/PyCharm/YuHangProjects/DROO/data/data_%d' %N)['input_h']
    rate = sio.loadmat('/home/yuhang/PyCharm/YuHangProjects/DROO/data/data_%d' %N)['output_obj'] # this rate is only used to plot figures; never used to train DROO.

    # increase h to close to 1 for better training; it is a trick widely adopted in deep learning
    channel = channel * 1000000

    # generate the train and test data sample index
    # data are splitted as 80:20
    # training data are randomly sampled with duplication if n > total data size

    split_idx = int(.8 * len(channel))
    num_test = min(len(channel) - split_idx, n - int(.8 * n)) # training data size

    ddpg = ddpgy.DDPG(state_dim=N,
                     action_dim=1,
                     action_bound=2**N,
                     replacement=REPLACEMENT,
                     memory_capacity=Memory)


    start_time = time.time()

    rate_his = []
    rate_his_ratio = []

    # i_idx = 500 % split_idx
    #
    # h = channel[i_idx, :]
    # print("h=", h.shape)
    # m = ddpg.choose_action(h)
    # print("mmm==", m)
    # m = str(bin(int(m*1024)-1))
    # m = m[2:]
    # M = np.array(list(m))
    # print("mmm==", m)
    # print("M==", m)
    # print("m_shape=", m.shape)


    for i in range(n):
        if i % (n//10) == 0:
           print("%0.1f"%(i/n))

        if i < n - num_test:
            # training
            i_idx = i % split_idx
        else:
            # test
            i_idx = i - n + num_test + split_idx

        h = channel[i_idx,:]

        # the action selection must be either 'OP' or 'KNN'
        if i <= np.random.uniform(0, 1):
            m = ddpg.choose_action(h)
        else:
            m = np.random.choice([0, 1])

        # x = str(bin(int(m*1024)-1))
        # x = x[2:]
        # X = np.array(list(map(int, x)))
        # print("X=",X)
        # print("X_shape", X.shape)

        x = bin(int(m*1023))
        x = list(x[2:])
        x = [int(u) for u in x]
        X = np.array(x)

        # print("X=",X)
        # print("X_shape", X.shape)


        r = bisection(h/1000000, X)[0]
        r_list = []
        r_list.append(r)
        # encode the mode with largest reward
        ddpg.encode(h, m)
        # the main code for DROO training ends here




        # the following codes store some interested metrics for illustrations
        # memorize the largest reward
        rate_his.append(np.max(r_list))
        rate_his_ratio.append(rate_his[-1] / rate[i_idx][0])


    total_time=time.time()-start_time
    # ddpg.plot_a_cost()
    # ddpg.plot_c_cost()
    plot_rate(rate_his_ratio)

    print('Total time consumed:%s'%total_time)
    print('Average time per channel:%s'%(total_time/n))


