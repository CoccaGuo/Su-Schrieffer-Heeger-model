import numpy as np
import matplotlib.pyplot as plt
from models import SSH

class SSHpsi:
    def __init__(self, length,pos,t1l=0.5, t1r=1.5,t2=1.0, gamma=0.):
        self.gamma = gamma
        self.t1l = t1l
        self.t1r = t1r
        self.t2 = t2
        self.N = length
        self.chain = SSH.SSH(length=length, t1rp=t1r + gamma, t1rm=t1r - gamma,t1lm=t1l+gamma,t1lp=t1l-gamma ,pos=pos, bloch=False)

if __name__ == '__main__':
    ssh = SSHpsi(length=20,pos= 10)
    _,wave = ssh.chain.wavefunction
    from matplotlib.pyplot import MultipleLocator
    x_major_locator = MultipleLocator(1)
    ax = plt.gca()
    ax.xaxis.set_major_locator(x_major_locator)
    # plt.bar(np.arange(1, 11, 0.5), (wave[:, 9]),color='pink',width=0.4)
    plt.bar(np.arange(1, 21, 0.5), (wave[:,19]), color='pink',width=0.4)
    # plt.bar(np.arange(1, 21), np.abs(wave[10, :]),color='red')
    # plt.bar(np.arange(1, 21), np.abs(wave[9, :]),color='dodgerblue')
    plt.ylim(-0.5,0.65)
    plt.xlim(0,20)
    plt.xlabel('cell index')
    plt.ylabel('wavefunction')
