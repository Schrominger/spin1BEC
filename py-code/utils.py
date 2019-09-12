#  -*- coding: utf-8 -*-

"""
file: utils.py
author: minger
date: 2019-09-08 
"""


import numpy as np
import scipy as sp
import qutip as qt
import matplotlib.pyplot as plt


def build_h0(N, M_arr=np.array([0,])):
    # 按照给定M顺序生存block矩阵的array组成的dict
    h0 = {}
    for M in M_arr:
        kmax = (N-abs(M))//2
        k_arr = np.arange(0., kmax+1.0) if M>=0 else np.arange(abs(M), (N+abs(M))//2+1.0)
        # q*(M+2k), neglect q here
        # h0[M] = np.diag(M+2.0*k_arr - N)  # -N0
        h0[M] = np.diag((2.0*k_arr+M)) # (N1 + N_{-1})
    return h0

def build_h2(N, M_arr=np.array([0,])):
    h2 = {}
    for M in M_arr:
        kmax = np.floor((N-np.abs(M))/2.)
        k_arr = np.arange(0, kmax+1.0) if M>= 0 else np.arange(abs(M), (N+abs(M))//2+1.0)
        # F^2-2N 
        d = (2.0*(N - M - 2.0*k_arr)-1.0)*(M + 2.0*k_arr) + M**2
        # off-diagonal term 
        if np.size(d)>1:
            up = 2.0*np.sqrt((N-2.0*k_arr-M+1.0)*(N-2.0*k_arr-M+2.))*np.sqrt(k_arr*(k_arr+M))
            up = up[1:]
            h2[M] = np.diag(d,0) + np.diag(up,1) + np.diag(up,-1)
        else:
            h2[M] = np.diag(d,0)

    return h2

# 只需传入M的序列，按照M的排序来顺序定，或者按照h0的尺寸来定
def build_h1(h0, N):
    # 实际上只需要h0的各个block的维度
    # only for Fx now
    h1 = {}
    # iterate the M values, only non-negtive.
    M_arr = [x for x in h0.keys() if not x<0]
    for M in M_arr[0:-1]: 
        (m,n) = (np.shape(h0[M])[0], np.shape(h0[M+1])[0])
        temp = np.full((m, n), 0.0)
        # 赋值： diagonal elements
        id_arr = np.arange(0,min(m,n)) # 对角元指标

        temp[id_arr, id_arr] = (1.0/np.sqrt(2))*np.sqrt((N-2.*id_arr-M)*(id_arr+M+1.))
        # up-diagonal elements
        up_arr = np.arange(1,n) # +1 off-diagonal line has n-1 elements
        temp[up_arr,up_arr-1] = (1.0/np.sqrt(2))*np.sqrt((N-2.*up_arr-M+1.)*(up_arr))
        
        h1[M+1] = temp[:,:]
        h1[-M-1] = temp[:,:] 

    return h1

# combine function， combine these blocks
def merge_h(h0, h1, h2, c=1.0, q=1.0, r=0.0):
    """
    merge all h-components
    h = q*h0 + (c/2N)*h2 + r*h1
    """
    H_dim = int(np.sum([np.floor((N-np.abs(Fz))//2)+1 for Fz in h0.keys()]))
    print("The full Hilber dim=", H_dim,".")
    fullH = np.full((H_dim, H_dim), 0.0)
    
    cnt = 0
    for M in sorted(h0): 
        h1_Mid = M if M<0 else M+1
        if h1_Mid in h1.keys(): # h1 has less one key than h0 and h2.
            # block_h0, block_h2, block_h1 = h0[M], h2[M],h1[M]
            current_shape, next_shape = np.shape(h0[M])[0], np.shape(h0[M+1])[0] 
            # print('M:',M)
            # pdb.set_trace()
            # diagonal block, black tri-diagonal matrix.
            fullH[cnt:cnt+current_shape, cnt:cnt+current_shape] \
                                           = q*h0[M][:,:] + (c/N/2.0)*h2[M][:,:]
            # off-diagonal block, (blue rectangular array)
            fullH[cnt:cnt+current_shape, cnt+current_shape:cnt+current_shape+next_shape] \
                                           = r*np.transpose(h1[h1_Mid][:,:]) if h1_Mid<0 else r*h1[h1_Mid][:,:] # +1 off-diagonal
            fullH[cnt+current_shape:cnt+current_shape+next_shape, cnt:cnt+current_shape] \
                                           = r*h1[h1_Mid][:,:] if h1_Mid<0 else r*np.transpose(h1[h1_Mid][:,:]) # -1 off-diagonal

            cnt += current_shape
        else:
            current_shape = np.shape(h0[M])[0]
            # diagonal block, black tri-diagonal matrix.
            fullH[cnt:cnt+current_shape, cnt:cnt+current_shape] = q*h0[M][:,:] + (c/2./N)*h2[M][:,:]
            # no blue rectangular block for this M-value.
            cnt += current_shape

    return fullH


if __name__ == "__main__":
    import pdb
    N = 10
    M_arr = np.arange(-N,N+1)
    print('N=', N)
    h0 = build_h0(N, M_arr=M_arr)
    h2 = build_h2(N, M_arr=M_arr)

    h1 = build_h1(h0, N)
    # pdb.set_trace()
    H = merge_h(h0, h1, h2, c=1.0, q=1.0, r=0.00)
    Hobj = qt.Qobj(H)
    # pdb.set_trace()
    e = Hobj.eigenenergies()
    print(e[0:30]+1.0)
    # Gap = []
    # qvec = np.linspace(-2.2,2.2,200)
    # for q in qvec:
    #     H = merge_h(h0, h1, h2, q=q)
    #     Hobj = qt.Qobj(H)
    #     e = Hobj.eigenenergies()
    #     gap = e[1] - e[0]
    #     Gap.append(gap)
    # plt.plot(qvec, Gap)
    # plt.show()
    # pdb.set_trace()

