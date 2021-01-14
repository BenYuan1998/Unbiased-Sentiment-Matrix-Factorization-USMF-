# -*- coding: utf-8 -*-
"""
Created on Wed Dec 30 17:58:38 2020

@author: Administrator
"""
import numpy as np
def checking_convergence(iterations,costs):
    import matplotlib.pyplot as plt
    plt.plot(iterations,costs)
    plt.xlabel("ith iteration")
    plt.ylabel("value of the loss function")
    plt.show()
def trainingalgorithm_EFM(A, X, Y, r, r_, lambda_x, lambda_y, lambda_u, lambda_h, lambda_v, T=500, alpha=1e-3):
    m = X.shape[0]
    p = X.shape[1]
    n = Y.shape[0]
    U1 = np.random.rand(m, r)
    U2 = np.random.rand(n, r)
    V = np.random.rand(p, r)
    H1 = np.random.rand(m, r_)
    H2 = np.random.rand(n, r_)
    t = 0
    iterations=list()
    costs=list()
    while t < T:
        t += 1
        iterations.append(t)
        _U1 = U1
        _U2 = U2
        _V = V
        _H1 = H1
        _H2 = H2
        E = 0
        for i in range(len(A)):
            for j in range(len(A[i])):
                if A[i, j] > 0:
                    e1_ij = A[i,j] - U1[i, :].dot(U2.T[:, j]) - H1[i, :].dot(H2.T[:, j])
                    E += pow(e1_ij, 2)
                    for k in range(r):
                        _U1[i, k] = U1[i, k] + alpha * (2 * e1_ij * U2[j, k] - 2 * lambda_u * U1[i, k])
                        _U2[j, k] = U2[j, k] + alpha * (2 * e1_ij * U1[i, k] - 2 * lambda_u * U2[j, k])
                        #E += lambda_u * (pow(U1[i, k], 2) + pow(U2[j, k], 2))
                    for k in range(r_):
                        _H1[i, k] = H1[i, k] + alpha * (2 * e1_ij * H2[j, k] - 2 * lambda_h * H1[i, k])
                        _H2[j, k] = H2[j, k] + alpha * (2 * e1_ij * H1[i, k] - 2 * lambda_h * H2[j, k])
                        #E += lambda_h * (pow(H1[i, k], 2) + pow(H2[j, k], 2))
        for i in range(len(X)):
            for j in range(len(X[i])):
                if X[i, j] > 0:
                    e2_ij = X[i, j] - U1[i, :].dot(V.T[:, j])
                    E += pow(e2_ij, 2)
                    for k in range(r):
                        _U1[i, k] = _U1[i, k] +alpha * (2 * e2_ij * V[j, k])
                        _V[j, k] = V[j, k] + alpha * (2 * e2_ij * U1[i, k] - 2 * lambda_v * V[j, k])
                        #E += lambda_v * (pow(V[j, k], 2))
        for i in range(len(Y)):
            for j in range(len(Y[i])):
                if Y[i, j] > 0:
                    e3_ij = Y[i, j] - U2[i, :].dot(V.T[:, j])
                    E += pow(e3_ij, 2)
                    for k in range(r):
                        _U2[i, k] = _U2[i, k] + alpha * (2 * e3_ij * V[j, k])
                        _V[j, k] = _V[j, k] + alpha * (2 * e3_ij * U2[i, k])
        e4 = lambda_u * (np.sum(U1 ** 2) + np.sum(U2 ** 2))
        e5 = lambda_h * (np.sum(H1 ** 2) + np.sum(H2 ** 2))
        e6 = lambda_v * (np.sum(V ** 2))
        #print ("E: " + str(E))
        E += e4 + e5 + e6
        costs.append(E)
        #print E
        U1 = _U1
        U2 = _U2
        V = _V
        H1 = _H1
        H2 = _H2
    return [U1, U2, V, H1, H2,iterations,costs]

