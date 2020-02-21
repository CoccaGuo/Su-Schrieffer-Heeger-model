import numpy as np
import matplotlib.pyplot as plt
from models import SSH

plt.rcParams['figure.dpi'] = 300  # dpi of the pictures

if __name__ == '__main__':
    gamma = 0
    t1l = -0.1
    t1r = 1.5
    t1lp = t1l - 0.5 * gamma
    t1lm = t1l + 0.5 * gamma
    t1rp = t1r - 0.5 * gamma
    t1rm = t1r + 0.5 * gamma
    chain = SSH.SSH(length=80,t1lp=t1lp,t1lm=t1lm,t1rp=t1rp,t1rm=t1rm,pos=40)
    energy, vector = chain.wavefunction
    vector_a = vector[::2,:]
    vector_b = vector[:, 1::2]
    plt.plot(np.abs(vector_a), 'k')
    plt.plot(np.abs(vector_a[:,0]), 'r')
    plt.show()





