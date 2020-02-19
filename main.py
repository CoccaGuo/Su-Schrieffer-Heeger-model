#!main function
import sys

sys.path.append('./models')
import numpy as np
import matplotlib.pyplot as plt
import SSH

if __name__ == '__main__':
    t1 = 0.5
    t2 = 1
    L = 40
    delta = 2
    pos = 20
    h_dw = SSH.H_domain_wall_delta_hopping1(t1, t2, L, delta, pos)
    ener, vect = np.linalg.eig(h_dw)
    for ind in range(0, L):
        plt.plot(vect[:, ind]**2,'k')
    plt.show()
